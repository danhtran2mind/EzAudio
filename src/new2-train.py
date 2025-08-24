import random
import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from diffusers import DDIMScheduler

from models.udit import RMSNorm
from models.conditioners import MaskDiT
from modules.autoencoder_wrapper import Autoencoder
from transformers import T5Tokenizer, T5EncoderModel
from dataset.audiocaps_v2 import EACaps
from utils import scale_shift, get_lr_scheduler, compute_snr, load_yaml_with_includes

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
try:
    import wandb
except ImportError:
    wandb = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------------------------- #
#                        Directory Setup Function                            #
# -------------------------------------------------------------------------- #
def setup_directories(args, params):
    """Create directories for saving model checkpoints and logs."""
    args.save_dir = os.path.join(args.save_dir, params['model_name']) + '/'
    args.log_dir = os.path.join(args.save_dir, 'logs') + '/'
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)


# -------------------------------------------------------------------------- #
#                        Device Configuration Function                        #
# -------------------------------------------------------------------------- #
def set_device(args):
    """Configure device (CPU/GPU) and set random seeds for reproducibility."""
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'


# -------------------------------------------------------------------------- #
#                        Optimizer Setup Function                            #
# -------------------------------------------------------------------------- #
def setup_optimizer(unet, params):
    """Configure AdamW optimizer with separate decay/no-decay parameter groups."""
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, RMSNorm)
    no_decay_suffixes = ['bias', 'abs_pe', 'alpha', 'beta', 'mask_embed', 'scale_shift_table', 'cfg_embedding']
    for mn, m in unet.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn
            if any(pn.endswith(suffix) for suffix in no_decay_suffixes):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in unet.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"Parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": params['opt']['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=params['opt']['learning_rate'], 
                                  betas=(params['opt']['beta1'], params['opt']['beta2']), 
                                  eps=params['opt']['adam_epsilon'])
    return optimizer


# -------------------------------------------------------------------------- #
#                        Batch Preparation Functions                         #
# -------------------------------------------------------------------------- #
def prepare_batch(args, batch, autoencoder, tokenizer, text_encoder, params):
    """Prepare a batch for training with audio encoding and optional text processing."""
    audio_clip, text_batch = batch
    with torch.no_grad():
        audio_clip = autoencoder(audio=audio_clip)
        if tokenizer is not None:
            text_batch_np = np.array(text_batch)
            cfg_mask = torch.rand(len(text_batch_np)) < params['text_encoder']['cfg']
            text_batch_np[cfg_mask] = ""
            text_batch = text_batch_np.tolist()
            text_batch = tokenizer(text_batch, max_length=params['text_encoder']['max_length'], 
                                   padding="max_length", truncation=True, return_tensors="pt")
            text_mask = text_batch.attention_mask.to(audio_clip.device).bool()
            text = text_encoder(input_ids=text_batch.input_ids.to(audio_clip.device),
                                attention_mask=text_mask).last_hidden_state
        else:
            text, text_mask = None, None
    return audio_clip, text, text_mask


def prepare_batch_cache(args, batch, autoencoder, tokenizer, text_encoder, params):
    """Prepare a batch with cached text embeddings for offline processing."""
    audio_clip, text, text_mask = batch
    with torch.no_grad():
        audio_clip = autoencoder(audio=audio_clip)
    if tokenizer is None:
        text, text_mask = None, None
    return audio_clip, text, text_mask


# -------------------------------------------------------------------------- #
#                          Loss Computation Function                         #
# -------------------------------------------------------------------------- #
def compute_loss(model_pred, target, mask, noise_scheduler, timesteps, snr_gamma=None):
    """Compute the loss for diffusion model training, optionally weighted by SNR."""
    if snr_gamma is None:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss * mask.float()
        loss = loss.sum(dim=[1, 2]) / mask.sum(dim=[1, 2])
        loss = loss.mean()
    else:
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)
        else:
            raise NotImplementedError
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss * mask.float()
        loss = loss.sum(dim=[1, 2]) / mask.sum(dim=[1, 2])
        loss = loss * mse_loss_weights
        loss = loss.mean()
    return loss


# -------------------------------------------------------------------------- #
#                          Validation Function                               #
# -------------------------------------------------------------------------- #
def validate(unet, val_loader, autoencoder, tokenizer, text_encoder, noise_scheduler, params, accelerator, args):
    """Evaluate the model on the validation set and compute average loss."""
    unet.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if args.offline:
                audio_clip, text, text_mask = prepare_batch_cache(args, batch, autoencoder, tokenizer, text_encoder, params)
            else:
                audio_clip, text, text_mask = prepare_batch(args, batch, autoencoder, tokenizer, text_encoder, params)
            audio_clip = scale_shift(audio_clip, params['autoencoder']['scale'], params['autoencoder']['shift'])
            audio_clip = audio_clip[:, :, :params['data']['train_frames']]
            noise = torch.randn(audio_clip.shape).to(accelerator.device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (noise.shape[0],), device=accelerator.device).long()
            noisy_target = noise_scheduler.add_noise(audio_clip, noise, timesteps)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                velocity = noise_scheduler.get_velocity(audio_clip, noise, timesteps)
                target = velocity
            pred, mask = unet(noisy_target, timesteps, text, context_mask=text_mask, cls_token=None, gt=audio_clip)
            loss = compute_loss(pred, target, mask, noise_scheduler, timesteps, snr_gamma=params['opt']['snr_gamma'])
            val_loss += loss.item()
            val_steps += 1
    return val_loss / val_steps if val_steps > 0 else 0.0
    

# -------------------------------------------------------------------------- #
#                    Refactored Training-Related Functions                   #
# -------------------------------------------------------------------------- #
def initialize_logging(args, params, accelerator):
    """Set up logging based on the specified reporting method."""
    if not accelerator.is_main_process:
        return None
    setup_directories(args, params)
    print(args)
    if args.report_to == 'tensorboard' and SummaryWriter is not None:
        return SummaryWriter(log_dir=args.log_dir)
    elif args.report_to == 'wandb' and wandb is not None:
        wandb.init(project="ezaudio_training", dir=args.log_dir, config=vars(args), resume="allow" if args.resume_from_checkpoint else None)
        return wandb
    return None


def close_logging(writer, args):
    """Close logging resources cleanly."""
    if args.report_to == 'tensorboard' and SummaryWriter is not None:
        writer.close()
    elif args.report_to == 'wandb' and wandb is not None:
        wandb.finish()


def log_training_progress(args, global_step, epoch, losses, lr, log_file, writer, accelerator):
    """Log training metrics at specified intervals."""
    if not accelerator.is_main_process or global_step % args.log_step != 0:
        return losses
    current_time = time.asctime(time.localtime(time.time()))
    epoch_info = f'Epoch: [{epoch + 1}][{args.epochs}]'
    batch_info = f'Global Step: {global_step:.1f}'
    loss_info = f'Train Loss: {losses / args.log_step:.6f}'
    lr_info = f'Learning Rate: {lr:.6f}'
    log_message = f'\n{current_time}\n{epoch_info}    {batch_info}    {loss_info}    {lr_info}\n'
    
    if args.report_to == 'tensorboard' and SummaryWriter is not None:
        writer.add_scalar('Loss/train', losses / args.log_step, global_step)
        writer.add_scalar('Learning_Rate', lr, global_step)
    elif args.report_to == 'wandb' and wandb is not None:
        wandb.log({'train_loss': losses / args.log_step, 'learning_rate': lr, 'global_step': global_step})
    elif args.report_to == 'log_file':
        with open(log_file, mode='a') as f:
            f.write(log_message)
    print(log_message)
    return 0.0  # Reset losses


def log_validation_progress(args, global_step, epoch, val_loss, log_file, writer, accelerator):
    """Log validation metrics at specified intervals."""
    if not accelerator.is_main_process:
        return
    current_time = time.asctime(time.localtime(time.time()))
    epoch_info = f'Epoch: [{epoch + 1}][{args.epochs}]'
    batch_info = f'Global Step: {global_step:.1f}'
    val_log_message = f'\n{current_time}\n{epoch_info}    {batch_info}    Validation Loss: {val_loss:.6f}\n'
    
    if args.report_to == 'tensorboard' and SummaryWriter is not None:
        writer.add_scalar('Loss/validation', val_loss, global_step)
    elif args.report_to == 'wandb' and wandb is not None:
        wandb.log({'val_loss': val_loss, 'global_step': global_step})
    elif args.report_to == 'log_file':
        with open(log_file, mode='a') as f:
            f.write(val_log_message)
    print(val_log_message)


def save_checkpoint(unet, optimizer, lr_scheduler, global_step, epoch, args, accelerator, best_loss, val_loss, accumulation_steps):
    """Save model checkpoint, training metadata, and best model based on validation loss."""
    if not accelerator.is_main_process:
        return best_loss
    # Save checkpoint at each save_step (adjusted for accumulation steps)
    step = int(global_step)  # Convert global_step to integer for filename
    if (step) % (args.save_step // accumulation_steps) == 0 and step > 0:
        ckpt_file_path = os.path.join(args.save_dir, f"step_{step}.pt")
        unwrapped_unet = accelerator.unwrap_model(unet)
        accelerator.save({
            "model": unwrapped_unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "best_loss": best_loss
        }, ckpt_file_path)
        print(f"\nModel checkpoint successfully saved to: {ckpt_file_path}")
    
        # Save as best.pt if validation loss improved
        if val_loss is not None and (best_loss is None or val_loss < best_loss):
            best_loss = val_loss
            best_file_path = os.path.join(args.save_dir, "best.pt")
            accelerator.save({
                "model": unwrapped_unet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "best_loss": best_loss
            }, best_file_path)
            print(f"\nBest model checkpoint saved to: {best_file_path}")
    
    return best_loss


def save_epoch_checkpoint(unet, global_step, epoch, args, accelerator):
    """Save model checkpoint at the end of an epoch."""
    if not accelerator.is_main_process:
        return
    ckpt_file_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
    unwrapped_unet = accelerator.unwrap_model(unet)
    accelerator.save({"model": unwrapped_unet.state_dict()}, ckpt_file_path)
    metadata_file = os.path.join(args.save_dir, 'training_metadata.pt')
    accelerator.save({"global_step": global_step, "epoch": epoch + 1}, metadata_file)
    print(f"\nEpoch {epoch+1} checkpoint successfully saved to: {ckpt_file_path}")


def process_training_step(unet, batch, autoencoder, tokenizer, text_encoder, noise_scheduler, optimizer, lr_scheduler, params, args, accumulation_steps, accelerator):
    """Process a single training step, including forward pass and backpropagation."""
    with accelerator.accumulate(unet):
        if args.offline:
            audio_clip, text, text_mask = prepare_batch_cache(args, batch, autoencoder, tokenizer, text_encoder, params)
        else:
            audio_clip, text, text_mask = prepare_batch(args, batch, autoencoder, tokenizer, text_encoder, params)
        audio_clip = scale_shift(audio_clip, params['autoencoder']['scale'], params['autoencoder']['shift'])
        audio_clip = audio_clip[:, :, :params['data']['train_frames']]
        noise = torch.randn(audio_clip.shape).to(accelerator.device)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (noise.shape[0],), device=accelerator.device).long()
        noisy_target = noise_scheduler.add_noise(audio_clip, noise, timesteps)
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(audio_clip, noise, timesteps)
        pred, mask = unet(noisy_target, timesteps, text, context_mask=text_mask, cls_token=None, gt=audio_clip)
        loss = compute_loss(pred, target, mask, noise_scheduler, timesteps, snr_gamma=params['opt']['snr_gamma'])
        accelerator.backward(loss)
        if accelerator.sync_gradients and params['opt'].get('grad_clip', 0) > 0:
            accelerator.clip_grad_norm_(unet.parameters(), max_norm=params['opt']['grad_clip'])
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    return loss.item() / accumulation_steps


def train(unet, train_loader, val_loader, autoencoder, tokenizer, text_encoder, 
          noise_scheduler, optimizer, lr_scheduler, accelerator, args, params):
    """Main training loop with modularized components."""
    global_step, start_epoch, best_loss = load_checkpoint(unet, optimizer, lr_scheduler, args, accelerator)
    accumulation_steps = args.accumulation_steps if args.accumulation_steps is not None else params['opt']['accumulation_steps']
    losses = 0.0
    log_file = os.path.join(args.log_dir, 'training_log.txt') if args.report_to == 'log_file' else None
    writer = initialize_logging(args, params, accelerator)
    
    accelerator.wait_for_everyone()
    for epoch in range(start_epoch, args.epochs):
        unet.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            loss = process_training_step(unet, batch, autoencoder, tokenizer, text_encoder, 
                                        noise_scheduler, optimizer, lr_scheduler, params, args, accumulation_steps, accelerator)
            global_step += 1 / accumulation_steps
            losses += loss
            if global_step % args.log_step == 0:
                lr = optimizer.param_groups[0]['lr']
                losses = log_training_progress(args, global_step, epoch, losses, lr, log_file, writer, accelerator)
            if global_step % args.val_step == 0 and global_step > 0:
                val_loss = validate(unet, val_loader, autoencoder, tokenizer, text_encoder, noise_scheduler, params, accelerator, args)
                log_validation_progress(args, global_step, epoch, val_loss, log_file, writer, accelerator)
                best_loss = save_checkpoint(unet, optimizer, lr_scheduler, global_step, epoch, args, accelerator, best_loss, val_loss, accumulation_steps)
                accelerator.wait_for_everyone()
                unet.train()
            elif (global_step) % (args.save_step // accumulation_steps) == 0 and global_step > 0:
                best_loss = save_checkpoint(unet, optimizer, lr_scheduler, global_step, epoch, args, accelerator, best_loss, None, accumulation_steps)
                accelerator.wait_for_everyone()
                unet.train()
            
            # Check if max_step is reached
            if args.max_step is not None and global_step >= args.max_step:
                print(f"Reached maximum step {args.max_step}. Stopping training.")
                save_epoch_checkpoint(unet, global_step, epoch, args, accelerator)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    close_logging(writer, args)
                return
    
    # Save final epoch checkpoint when training completes
    save_epoch_checkpoint(unet, global_step, epoch, args, accelerator)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        close_logging(writer, args)


def setup_dataset_and_loaders(args, params):
    """Initialize datasets and data loaders for training and validation."""
    train_set = EACaps(**params['data']['train'])
    if train_set.text_path is not None:
        args.offline = True
        print('Offline text embedding')
        t5_device = 'cpu'
    else:
        args.offline = False
        t5_device = args.device
    batch_size = args.batch_size if args.batch_size is not None else params['opt']['batch_size']
    train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=batch_size, shuffle=True)
    val_set = EACaps(**params['data'].get('val', params['data']['train']))
    val_loader = DataLoader(val_set, num_workers=args.num_workers, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, t5_device


def setup_models(args, params, t5_device, accelerator):
    """Initialize autoencoder, tokenizer, text encoder, and UNet models."""
    autoencoder = Autoencoder(ckpt_path=params['autoencoder']['path'], model_type=params['autoencoder']['name'], quantization_first=params['autoencoder']['q_first'])
    autoencoder.to(accelerator.device)
    autoencoder.eval()
    
    if args.stage == 'audioset':
        tokenizer = None
        text_encoder = None
    elif args.stage == 'audiocaps':
        tokenizer = T5Tokenizer.from_pretrained(params['text_encoder']['model'])
        text_encoder = T5EncoderModel.from_pretrained(params['text_encoder']['model'], device_map='cpu').to(t5_device)
        text_encoder.eval()
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")
    
    unet = MaskDiT(**params['model']).to(accelerator.device)
    return autoencoder, tokenizer, text_encoder, unet


def load_checkpoint(unet, optimizer, lr_scheduler, args, accelerator):
    """Load model checkpoint, optimizer, scheduler, and training state from the latest step_{int(step)}.pt."""
    global_step = 0
    start_epoch = 0
    best_loss = None
    if args.resume_from_checkpoint:
        # Find the latest step_{int(step)}.pt file
        checkpoint_dir = args.resume_from_checkpoint
        step_files = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
        if step_files:
            latest_step_file = max(step_files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            step_number = int(os.path.basename(latest_step_file).split('_')[1].split('.')[0])
            print(f"Loading latest checkpoint from: {latest_step_file}")
            # Load checkpoint
            if os.path.exists(latest_step_file):
                checkpoint = torch.load(latest_step_file, map_location='cpu')
                unet.load_state_dict(checkpoint['model'], strict=args.strict)
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                global_step = checkpoint.get('global_step', 0)
                start_epoch = checkpoint.get('epoch', 0)
                best_loss = checkpoint.get('best_loss', None)
                if accelerator.is_main_process:
                    total_params = sum([param.nelement() for param in unet.parameters()])
                    print(f"Resuming training from epoch {start_epoch}, global step {global_step}, best loss {best_loss}")
                    print("Number of parameter: %.2fM" % (total_params / 1e6))
        else:
            print(f"No step_*.pt files found in {checkpoint_dir}. Starting training from scratch.")
    elif args.ckpt:
        print(f"Loading model weights from: {args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location='cpu')['model']
        result = unet.load_state_dict(state_dict, strict=args.strict)
        if accelerator.is_main_process:
            if result.missing_keys:
                print("Warning: The following layers were not loaded because they are missing in the checkpoint:")
                for key in result.missing_keys:
                    print(f" - {key}")
            if result.unexpected_keys:
                print("Warning: The following layers were not expected in the model and thus were not loaded:")
                for key in result.unexpected_keys:
                    print(f" - {key}")
            total_params = sum([param.nelement() for param in unet.parameters()])
            print("Number of parameter: %.2fM" % (total_params / 1e6))
    return global_step, start_epoch, best_loss


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', type=str, default='src/configs/ezaudio-l.yml')
    parser.add_argument("--amp", type=str, default='fp16')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--save-step', type=int, default=500, help='Steps between saving checkpoints (adjusted for accumulation_steps)')
    parser.add_argument('--val-step', type=int, default=500, help='Steps between validation runs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for training and validation (overrides config if set)')
    parser.add_argument('--accumulation_steps', type=int, default=None, help='Number of gradient accumulation steps (overrides config if set)')
    parser.add_argument('--max_step', type=int, default=None, help='Maximum number of training steps (adjusted for accumulation_steps)')
    parser.add_argument('--random-seed', type=int, default=2024)
    parser.add_argument('--log-step', type=int, default=100)
    parser.add_argument('--report-to', type=str, default='none', choices=['tensorboard', 'log_file', 'wandb', 'none'], 
                        help='Logging method')
    parser.add_argument('--save-dir', type=str, default='./ckpts/')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--strict', type=bool, default=False)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint directory to resume training from')
    return parser.parse_args()


def main():
    """Main function to orchestrate training setup and execution."""
    args = parse_args()
    params = load_yaml_with_includes(args.config_name)
    args.stage = 'audioset' if params['model']['context_dim'] is None else 'audiocaps'
    args.mae = args.stage == 'audioset'
    set_device(args)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    accumulation_steps = args.accumulation_steps if args.accumulation_steps is not None else params['opt']['accumulation_steps']
    accelerator = Accelerator(mixed_precision=args.amp, gradient_accumulation_steps=accumulation_steps)
    train_loader, val_loader, t5_device = setup_dataset_and_loaders(args, params)
    autoencoder, tokenizer, text_encoder, unet = setup_models(args, params, t5_device, accelerator)
    noise_scheduler = DDIMScheduler(**params['diff'])
    optimizer = setup_optimizer(unet, params)
    lr_scheduler = get_lr_scheduler(optimizer, 'customized', warmup_steps=params['opt']['warmup'])
    unet, autoencoder, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        unet, autoencoder, optimizer, lr_scheduler, train_loader, val_loader
    )
    train(unet, train_loader, val_loader, autoencoder, tokenizer, text_encoder, 
          noise_scheduler, optimizer, lr_scheduler, accelerator, args, params)


if __name__ == '__main__':
    main()
