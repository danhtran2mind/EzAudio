import random
import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import glob
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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
#                        Device Configuration Function                       #
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
def validate(unet, val_loader, autoencoder, tokenizer, text_encoder, noise_scheduler, 
            params, accelerator, args):
    """Evaluate the model on the validation set and compute average loss."""
    unet.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if args.offline:
                audio_clip, text, text_mask = prepare_batch_cache(args, batch, autoencoder, 
                                                                  tokenizer, text_encoder, params)
            else:
                audio_clip, text, text_mask = prepare_batch(args, batch, autoencoder, 
                                                            tokenizer, text_encoder, params)
            audio_clip = scale_shift(audio_clip, params['autoencoder']['scale'], 
                                     params['autoencoder']['shift'])
            audio_clip = audio_clip[:, :, :params['data']['train_frames']]
            noise = torch.randn(audio_clip.shape).to(accelerator.device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                     (noise.shape[0],), device=accelerator.device).long()
            noisy_target = noise_scheduler.add_noise(audio_clip, noise, timesteps)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                velocity = noise_scheduler.get_velocity(audio_clip, noise, timesteps)
                target = velocity
            pred, mask = unet(noisy_target, timesteps, text, context_mask=text_mask, 
                              cls_token=None, gt=audio_clip)
            loss = compute_loss(pred, target, mask, noise_scheduler, timesteps, 
                                snr_gamma=params['opt']['snr_gamma'])
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
        wandb.init(
            project="ezaudio_training",
            dir=args.log_dir,
            config=vars(args),
            resume="allow" if args.resume_from_checkpoint else None
        )
        return wandb
    return None

def close_logging(writer, args):
    """Close logging resources cleanly."""
    if args.report_to == 'tensorboard' and SummaryWriter is not None:
        writer.close()
    elif args.report_to == 'wandb' and wandb is not None:
        wandb.finish()

def log_training_progress(args, global_step, epoch, losses, lr, log_file, 
                          writer, accelerator, accumulation_steps):
    """Log training metrics at specified intervals."""
    if not accelerator.is_main_process:
        return losses
    total_batches = global_step * accumulation_steps
    if total_batches % args.log_step != 0:
        return losses
    current_time = time.asctime(time.localtime(time.time()))
    epoch_info = f'Epoch: {epoch + 1}'
    batch_info = f'Global Step: {global_step}'
    loss_info = f'Train Loss: {losses / args.log_step:.6f}'
    lr_info = f'Learning Rate: {lr:.6f}'
    log_message = (f'\n{current_time}\n{epoch_info}    {batch_info}    '
                    f'{loss_info}    {lr_info}\n')
    
    if args.report_to == 'tensorboard' and SummaryWriter is not None:
        writer.add_scalar('Loss/train', losses / args.log_step, global_step)
        writer.add_scalar('Learning_Rate', lr, global_step)
    elif args.report_to == 'wandb' and wandb is not None:
        wandb.log({
            "train_loss": losses / args.log_step,
            "learning_rate": lr,
            "global_step": global_step
        })
    elif args.report_to == 'log_file':
        with open(log_file, mode='a') as f:
            f.write(log_message)
    print(log_message)
    return 0.0  # Reset losses

def log_validation_progress(args, global_step, epoch, val_loss, log_file, 
                            writer, accelerator):
    """Log validation metrics at specified intervals."""
    if not accelerator.is_main_process:
        return
    current_time = time.asctime(time.localtime(time.time()))
    epoch_info = f'Epoch: {epoch + 1}'
    batch_info = f'Global Step: {global_step}'
    val_log_message = (f'\n{current_time}\n{epoch_info}    {batch_info}    '
                        f'Validation Loss: {val_loss:.6f}\n')
    
    if args.report_to == 'tensorboard' and SummaryWriter is not None:
        writer.add_scalar('Loss/validation', val_loss, global_step)
    elif args.report_to == 'wandb' and wandb is not None:
        wandb.log({'val_loss': val_loss, 'global_step': global_step})
    elif args.report_to == 'log_file':
        with open(log_file, mode='a') as f:
            f.write(val_log_message)
    print(val_log_message)

# -------------------------------------------------------------------------- #
#                        Checkpoint Saving Functions                         #
# -------------------------------------------------------------------------- #
def save_checkpoint(unet, optimizer, lr_scheduler, global_step, epoch, args, accelerator, 
                    best_loss, val_loss, accumulation_steps, train_batch_size):
    """Save model checkpoint with training state."""
    if accelerator.is_main_process:
        checkpoint = {
            'model': unet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'best_loss': best_loss,
            'batch_size': train_batch_size,
            'accumulation_steps': accumulation_steps
        }
        checkpoint_dir = args.save_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-{global_step}/model.pt')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at {checkpoint_path}")
        
        # Manage checkpoint limits
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*')), key=os.path.getmtime)
        if len(checkpoints) > args.max_num_checkpoints:
            for old_checkpoint in checkpoints[:len(checkpoints) - args.max_num_checkpoints]:
                shutil.rmtree(old_checkpoint, ignore_errors=True)
        
        # Update best loss if validation loss is provided and improved
        if val_loss is not None and (best_loss is None or val_loss < best_loss):

            best_checkpoint = {
                'model': unet.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
            }

            best_loss = val_loss
            best_path = os.path.join(checkpoint_dir, 'model.pt')
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save(best_checkpoint, best_path)
            print(f"Saved best checkpoint at {best_path}")
    
    return best_loss



def process_training_step(unet, batch, autoencoder, tokenizer, text_encoder, 
                          noise_scheduler, optimizer, lr_scheduler, params, args, 
                          accumulation_steps, accelerator):
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
        
        # Ensure loss is a tensor and valid for backward
        if not isinstance(loss, torch.Tensor):
            raise ValueError(f"Loss is not a tensor, got type {type(loss)} with value {loss}")
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected (nan/inf), skipping backward pass")
            return 0.0  # Skip this step to avoid crashing
        
        accelerator.backward(loss)
        if accelerator.sync_gradients and params['opt'].get('grad_clip', 0) > 0:
            accelerator.clip_grad_norm_(unet.parameters(), max_norm=params['opt']['grad_clip'])
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    return loss.item()

# -------------------------------------------------------------------------- #
#                             Training Function                              #
# -------------------------------------------------------------------------- #
def train(unet, train_loader, val_loader, autoencoder, tokenizer, text_encoder, 
          noise_scheduler, optimizer, lr_scheduler, accelerator, args, params,
          train_batch_size, accumulation_steps):
    """
    Training loop with resume support from global_step, start_epoch, and batch_idx.
    Uses gradient accumulation and supports logging, validation, and checkpointing.
    Saves best weights with best_loss using save_checkpoint at training completion.
    """
    # Load checkpoint
    global_step, start_epoch, best_loss, checkpoint_batch_size, checkpoint_accumulation_steps = load_checkpoint(unet, optimizer, lr_scheduler, args, accelerator)
    print("=" * 10, f"Resuming from step: {global_step}, epoch: {start_epoch}")
    print("args.save_step ", args.save_step)
    print("args.val_step ", args.val_step)
    print("Current train_batch_size:", train_batch_size)
    print("Current accumulation_steps:", accumulation_steps)
    print("Checkpoint batch_size:", checkpoint_batch_size)
    print("Checkpoint accumulation_steps:", checkpoint_accumulation_steps)

    losses = 0.0
    val_loss = None
    log_file = os.path.join(args.log_dir, 'training_log.txt') if args.report_to == 'log_file' else None
    writer = initialize_logging(args, params, accelerator)

    accelerator.wait_for_everyone()
    
    for epoch in range(start_epoch, args.epochs):
        unet.train()
        epoch_losses = 0.0
        epoch_steps = 0
        local_losses = []
        
        total_effective_steps = len(train_loader) // accumulation_steps + (1 if len(train_loader) % accumulation_steps else 0)
        # Calculate effective steps already processed in the current epoch
        if epoch == start_epoch and args.resume_from_checkpoint:
            checkpoint_batch_size = checkpoint_batch_size or train_batch_size  # Fallback to current if missing
            checkpoint_accumulation_steps = checkpoint_accumulation_steps or accumulation_steps
            prev_train_loader_len = 800 // checkpoint_batch_size  # Assuming 800 samples
            prev_total_effective_steps = prev_train_loader_len // checkpoint_accumulation_steps + (1 if prev_train_loader_len % checkpoint_accumulation_steps else 0)
            if start_epoch == 0:
                prev_progress_percentage = global_step / prev_total_effective_steps
            else:
                prev_progress_percentage = (global_step % prev_total_effective_steps) / prev_total_effective_steps
            effective_steps_done = int(prev_progress_percentage * total_effective_steps)
        else:
            effective_steps_done = 0
        remaining_effective_steps = total_effective_steps - effective_steps_done
        remaining_percentage = (remaining_effective_steps / total_effective_steps) * 100
        start_batch = effective_steps_done * accumulation_steps
        accelerator.print((f"Total effective steps for epoch {epoch + 1}: "
            f"{total_effective_steps}, Remaining: {remaining_effective_steps}, "
            f"Remaining Percentage: {remaining_percentage:.2f}%"))

        progress_bar = tqdm(total=total_effective_steps, 
            desc=f"Epoch {epoch + 1} (Effective Steps)", 
            initial=effective_steps_done,
            disable=not accelerator.is_main_process)

        for step, batch in enumerate(train_loader, start=start_batch):
            try:
                loss = process_training_step(
                    unet, batch, autoencoder, tokenizer, text_encoder,
                    noise_scheduler, optimizer, lr_scheduler, params, args,
                    accumulation_steps, accelerator
                )
                
                loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                local_losses.append(loss_value)
                epoch_losses += loss_value
                epoch_steps += 1
                
            except (RuntimeError, ValueError) as e:
                accelerator.print(f"Error in step {step}: {e}")
                continue
                
            is_accumulating = (step + 1) % accumulation_steps != 0
            is_last_step = (step + 1) == len(train_loader)
            do_optimization_step = not is_accumulating or is_last_step

            if do_optimization_step:
                avg_loss = sum(local_losses) / len(local_losses) if local_losses else 0.0
                losses += avg_loss
                local_losses = []
                global_step += 1
                progress_bar.update(1)
                
                if global_step % args.log_step == 0:
                    lr = optimizer.param_groups[0]['lr']
                    log_training_progress(args, global_step, epoch, losses, lr, log_file, writer, accelerator, accumulation_steps)
                    losses = 0.0
                    
                if args.val_step and global_step % args.val_step == 0:
                    val_loss = validate(unet, val_loader, autoencoder, tokenizer, text_encoder,
                                        noise_scheduler, params, accelerator, args)
                    log_validation_progress(args, global_step, epoch, val_loss, log_file, writer, accelerator)
                    unet.train()
                    
                if args.save_step and global_step % args.save_step == 0:
                    best_loss = save_checkpoint(unet, optimizer, lr_scheduler, global_step, epoch,
                                                args, accelerator, best_loss, val_loss, accumulation_steps, train_batch_size)
                    accelerator.wait_for_everyone()
                    unet.train()
                    
                if args.max_step is not None and global_step >= args.max_step:
                    accelerator.print(f"Reached max_step {args.max_step}. Saving final checkpoint and best epoch checkpoint.")
                    best_loss = save_checkpoint(unet, optimizer, lr_scheduler, global_step, epoch,
                                                args, accelerator, best_loss, val_loss, accumulation_steps, train_batch_size)
                    # save_epoch_checkpoint(unet, global_step, epoch, args, accelerator, accumulation_steps, train_batch_size, best_loss)
                    accelerator.wait_for_everyone()
                    progress_bar.close()
                    close_logging(writer, args)
                    return

        progress_bar.close()

    accelerator.print("Training complete. Saving final checkpoint and best epoch checkpoint.")
    best_loss = save_checkpoint(unet, optimizer, lr_scheduler, global_step, epoch,
                                args, accelerator, best_loss, val_loss, accumulation_steps, train_batch_size)
    # save_epoch_checkpoint(unet, global_step, epoch, args, accelerator, accumulation_steps, train_batch_size, best_loss)
    accelerator.wait_for_everyone()
    close_logging(writer, args)

# -------------------------------------------------------------------------- #
#                        Model Setup Functions                               #
# -------------------------------------------------------------------------- #
def setup_dataset_and_loaders(args, params, train_batch_size):
    """Initialize datasets and data loaders for training and validation."""
    # Initialize the full training dataset
    full_train_set = EACaps(**params['data']['train'])
    
    # Check text_path on the original dataset
    if full_train_set.text_path is not None:
        args.offline = True
        print('Offline text embedding')
        t5_device = 'cpu'
    else:
        args.offline = False
        t5_device = args.device
    
    # Limit train_set to 800 samples
    train_set_size = len(full_train_set)
    if train_set_size > 800:
        indices = list(range(800))  # Take first 800 samples
        train_set = Subset(full_train_set, indices)
        print(f"Training set limited to 800 samples (original size: {train_set_size})")
    else:
        train_set = full_train_set
        print(f"Training set has {train_set_size} samples, using all (no limiting needed)")

    # Use val_batch_size if provided, otherwise default to train_batch_size
    val_batch_size = args.val_batch_size if args.val_batch_size is not None else train_batch_size

    train_loader = DataLoader(train_set, num_workers=args.num_workers, 
                              batch_size=train_batch_size, shuffle=True)
    val_set = EACaps(**params['data'].get('val', params['data']['train']))
    val_loader = DataLoader(val_set, num_workers=args.num_workers, 
                            batch_size=val_batch_size, shuffle=False)
    print(f"train_loader length: {len(train_loader)} (batch_size={train_batch_size})")
    print(f"val_loader length: {len(val_loader)} (batch_size={val_batch_size})")
    return train_loader, val_loader, t5_device

def setup_models(args, params, t5_device, accelerator):
    """Initialize autoencoder, tokenizer, text encoder, and UNet models."""
    autoencoder = Autoencoder(ckpt_path=params['autoencoder']['path'], 
                              model_type=params['autoencoder']['name'], 
                              quantization_first=params['autoencoder']['q_first'])
    autoencoder.to(accelerator.device)
    autoencoder.eval()
    
    if args.stage == 'audioset':
        tokenizer = None
        text_encoder = None
    elif args.stage == 'audiocaps':
        tokenizer = T5Tokenizer.from_pretrained(params['text_encoder']['model'])
        text_encoder = T5EncoderModel.from_pretrained(params['text_encoder']['model'], 
                                                      device_map='cpu').to(t5_device)
        text_encoder.eval()
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")
    
    unet = MaskDiT(**params['model']).to(accelerator.device)
    return autoencoder, tokenizer, text_encoder, unet

# -------------------------------------------------------------------------- #
#                        Checkpoint Loading Function                         #
# -------------------------------------------------------------------------- #
def load_checkpoint(unet, optimizer, lr_scheduler, args, accelerator):
    """Load model checkpoint, optimizer, scheduler, and training state from the specified checkpoint file."""
    global_step = 0
    start_epoch = 0
    best_loss = None
    checkpoint_batch_size = None
    checkpoint_accumulation_steps = None
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Load model state
            unet.load_state_dict(checkpoint.get('model', {}), strict=args.strict)
            # Load optimizer and scheduler states if available
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # Load training metadata
            global_step = checkpoint.get('global_step', 0)
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('best_loss', None)
            checkpoint_batch_size = checkpoint.get('batch_size', None)
            checkpoint_accumulation_steps = checkpoint.get('accumulation_steps', None)
            if accelerator.is_main_process:
                total_params = sum([param.nelement() for param in unet.parameters()])
                print(f"Resuming training from epoch {start_epoch}, global step {global_step}, best loss {best_loss}")
                print("Number of parameter: %.2fM" % (total_params / 1e6))
                # Print warnings for missing or unexpected keys
                result = unet.load_state_dict(checkpoint.get('model', {}), strict=args.strict)
                if result.missing_keys:
                    print("Warning: The following layers were not loaded because they are missing in the checkpoint:")
                    for key in result.missing_keys:
                        print(f" - {key}")
                if result.unexpected_keys:
                    print("Warning: The following layers were not expected in the model and thus were not loaded:")
                    for key in result.unexpected_keys:
                        print(f" - {key}")
        else:
            print(f"Checkpoint file {checkpoint_path} not found. Starting training from scratch.")
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
            print("Number of parameter: %.2fM " % (total_params / 1e6))
    return (global_step, start_epoch, best_loss, checkpoint_batch_size, 
            checkpoint_accumulation_steps)

def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', type=str, default='src/configs/ezaudio-l.yml')
    parser.add_argument("--amp", type=str, default='fp16')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--save-step', type=int, default=500, 
                        help='Steps between saving checkpoints')
    parser.add_argument('--val-step', type=int, default=500, 
                        help='Steps between validation runs')
    parser.add_argument('--train_batch_size', type=int, default=2, 
                        help=('Batch size for training (overrides config if set)'))
    parser.add_argument('--val_batch_size', type=int, default=2, 
                        help=('Batch size for validation (overrides config if set; defaults to train-batch-size if not set)'))
    parser.add_argument('--accumulation_steps', type=int, default=2, 
                        help=('Number of gradient accumulation steps '
                              '(overrides config if set)'))
    parser.add_argument('--max-step', type=int, default=None, 
                        help='Maximum number of training steps')
    parser.add_argument('--max_num_checkpoints', type=int, default=3, 
                        help=('Maximum number of step-based checkpoints to keep '
                              '(excluding best.pt and epoch checkpoints)'))
    parser.add_argument('--random-seed', type=int, default=2024)
    parser.add_argument('--log-step', type=int, default=100)
    parser.add_argument('--report-to', type=str, default='none', 
                        choices=['tensorboard', 'log_file', 'wandb', 'none'], 
                        help='Logging method')
    parser.add_argument('--save-dir', type=str, default='ckpts/')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--strict', type=bool, default=False)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                        help='Path to a specific checkpoint file to resume training from')
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
    np.random.seed(args.random_seed)
    
    # Override batch size and accumulation steps
    train_batch_size = args.train_batch_size if args.train_batch_size is not None else params['opt'].get('batch_size', 16)
    accumulation_steps = args.accumulation_steps if args.accumulation_steps is not None else params['opt'].get('accumulation_steps', 1)
    
    accelerator = Accelerator(mixed_precision=args.amp, 
                              gradient_accumulation_steps=accumulation_steps)

    print("accumulation_steps: ", accumulation_steps)
    print("accelerator.gradient_accumulation_steps ", 
          accelerator.gradient_accumulation_steps)

    train_loader, val_loader, t5_device = setup_dataset_and_loaders(args, params, train_batch_size)
    autoencoder, tokenizer, text_encoder, unet = setup_models(args, params, t5_device, accelerator)
    noise_scheduler = DDIMScheduler(**params['diff'])
    optimizer = setup_optimizer(unet, params)
    lr_scheduler = get_lr_scheduler(optimizer, 'customized', warmup_steps=params['opt']['warmup'])
    unet, autoencoder, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        unet, autoencoder, optimizer, lr_scheduler, train_loader, val_loader
    )
    print("=" * 10, "train_loader: ", len(train_loader))
    print("=" * 10, "val_loader: ", len(val_loader))
    train(unet, train_loader, val_loader, autoencoder, tokenizer, text_encoder, 
          noise_scheduler, optimizer, lr_scheduler, accelerator, args, params,
          train_batch_size, accumulation_steps)

if __name__ == '__main__':
    main()
