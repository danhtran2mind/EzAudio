import random
import argparse
import os
import time
import numpy as np
from tqdm import tqdm

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

# Existing functions (unchanged for brevity)
# - setup_directories
# - set_device
# - setup_optimizer
# - prepare_batch
# - prepare_batch_cache
# - compute_loss
# - validate
# - load_checkpoint
# - parse_args
# (See original code for their implementations)

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

def save_checkpoint(unet, optimizer, lr_scheduler, global_step, epoch, args, accelerator):
    """Save model checkpoint and training metadata."""
    if not accelerator.is_main_process:
        return
    ckpt_file_path = os.path.join(args.save_dir, f"step_{global_step+1}.pt")
    unwrapped_unet = accelerator.unwrap_model(unet)
    accelerator.save({"model": unwrapped_unet.state_dict()}, ckpt_file_path)
    metadata_file = os.path.join(args.save_dir, 'training_metadata.pt')
    accelerator.save({"global_step": global_step, "epoch": epoch}, metadata_file)
    accelerator.save_state(os.path.join(args.save_dir, f"state_{global_step+1}"))
    print(f"\nModel checkpoint successfully saved to: {ckpt_file_path}")

def save_epoch_checkpoint(unet, global_step, epoch, args, accelerator):
    """Save model checkpoint at the end of an epoch."""
    if not accelerator.is_main_process:
        return
    ckpt_file_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
    unwrapped_unet = accelerator.unwrap_model(unet)
    accelerator.save({"model": unwrapped_unet.state_dict()}, ckpt_file_path)
    metadata_file = os.path.join(args.save_dir, 'training_metadata.pt')
    accelerator.save({"global_step": global_step, "epoch": epoch + 1}, metadata_file)
    accelerator.save_state(os.path.join(args.save_dir, f"state_epoch_{epoch+1}"))
    print(f"\nEpoch {epoch+1} checkpoint successfully saved to: {ckpt_file_path}")

def process_training_step(unet, batch, autoencoder, tokenizer, text_encoder, noise_scheduler, optimizer, lr_scheduler, params, args, accelerator):
    """Process a single training step, including forward pass and backpropagation."""
    with accelerator.accumulate(unet):
        # Prepare batch
        if args.offline:
            audio_clip, text, text_mask = prepare_batch_cache(args, batch, autoencoder, tokenizer, text_encoder, params)
        else:
            audio_clip, text, text_mask = prepare_batch(args, batch, autoencoder, tokenizer, text_encoder, params)
        # Scale and shift audio data
        audio_clip = scale_shift(audio_clip, params['autoencoder']['scale'], params['autoencoder']['shift'])
        audio_clip = audio_clip[:, :, :params['data']['train_frames']]
        # Add noise
        noise = torch.randn(audio_clip.shape).to(accelerator.device)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (noise.shape[0],), device=accelerator.device).long()
        noisy_target = noise_scheduler.add_noise(audio_clip, noise, timesteps)
        # Determine target
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(audio_clip, noise, timesteps)
        # Forward pass
        pred, mask = unet(noisy_target, timesteps, text, context_mask=text_mask, cls_token=None, gt=audio_clip)
        # Compute loss
        loss = compute_loss(pred, target, mask, noise_scheduler, timesteps, snr_gamma=params['opt']['snr_gamma'])
        # Backpropagation
        accelerator.backward(loss)
        if accelerator.sync_gradients and params['opt'].get('grad_clip', 0) > 0:
            accelerator.clip_grad_norm_(unet.parameters(), max_norm=params['opt']['grad_clip'])
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    return loss.item() / params['opt']['accumulation_steps']

def train(unet, train_loader, val_loader, autoencoder, tokenizer, text_encoder, 
          noise_scheduler, optimizer, lr_scheduler, accelerator, args, params):
    """Main training loop with modularized components."""
    global_step, start_epoch = load_checkpoint(unet, optimizer, lr_scheduler, args, accelerator)
    losses = 0.0
    log_file = os.path.join(args.log_dir, 'training_log.txt') if args.report_to == 'log_file' else None
    writer = initialize_logging(args, params, accelerator)
    
    accelerator.wait_for_everyone()
    for epoch in range(start_epoch, args.epochs):
        unet.train()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            loss = process_training_step(unet, batch, autoencoder, tokenizer, text_encoder, 
                                        noise_scheduler, optimizer, lr_scheduler, params, args, accelerator)
            global_step += 1 / params['opt']['accumulation_steps']
            losses += loss
            # Log progress
            if global_step % args.log_step == 0:
                lr = optimizer.param_groups[0]['lr']
                losses = log_training_progress(args, global_step, epoch, losses, lr, log_file, writer, accelerator)
            # Validate
            if global_step % args.val_step == 0 and global_step > 0:
                val_loss = validate(unet, val_loader, autoencoder, tokenizer, text_encoder, noise_scheduler, params, accelerator, args)
                log_validation_progress(args, global_step, epoch, val_loss, log_file, writer, accelerator)
            # Save checkpoint
            if (global_step + 1) % args.save_every_step == 0:
                save_checkpoint(unet, optimizer, lr_scheduler, global_step, epoch, args, accelerator)
                accelerator.wait_for_everyone()
                unet.train()
        
        # Save epoch checkpoint
        save_epoch_checkpoint(unet, global_step, epoch, args, accelerator)
        accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        close_logging(writer, args)

# -------------------------------------------------------------------------- #
#                          Main Execution Block                              #
# -------------------------------------------------------------------------- #
def main():
    """Main function to orchestrate training setup and execution."""
    # Parse arguments and load configuration
    args = parse_args()
    params = load_yaml_with_includes(args.config_name)
    
    # Determine training stage
    args.stage = 'audioset' if params['model']['context_dim'] is None else 'audiocaps'
    args.mae = args.stage == 'audioset'
    
    # Configure device and seeds
    set_device(args)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.amp, gradient_accumulation_steps=params['opt']['accumulation_steps'])
    
    # Setup datasets and loaders
    train_loader, val_loader, t5_device = setup_dataset_and_loaders(args, params)
    
    # Initialize models
    autoencoder, tokenizer, text_encoder, unet = setup_models(args, params, t5_device, accelerator)
    
    # Initialize noise scheduler
    noise_scheduler = DDIMScheduler(**params['diff'])
    
    # Setup optimizer and scheduler
    optimizer = setup_optimizer(unet, params)
    lr_scheduler = get_lr_scheduler(optimizer, 'customized', warmup_steps=params['opt']['warmup'])
    
    # Prepare for distributed training
    unet, autoencoder, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        unet, autoencoder, optimizer, lr_scheduler, train_loader, val_loader
    )
    
    # Start training
    train(unet, train_loader, val_loader, autoencoder, tokenizer, text_encoder, 
          noise_scheduler, optimizer, lr_scheduler, accelerator, args, params)

if __name__ == '__main__':
    main()
