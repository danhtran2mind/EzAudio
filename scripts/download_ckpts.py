import os
from huggingface_hub import hf_hub_download

configs = {'s3_xl': {'path': 'ckpts/s3/ezaudio_s3_xl.pt',
                     'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/s3/ezaudio_s3_xl.pt',
                     'config': 'ckpts/ezaudio-xl.yml'},
           's3_l': {'path': 'ckpts/s3/ezaudio_s3_l.pt',
                     'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/s3/ezaudio_s3_l.pt',
                     'config': 'ckpts/ezaudio-l.yml'},
          'vae': {'path': 'ckpts/vae/1m.pt', 
                  'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/vae/1m.pt'}
          }
def download_ckpt(model_dict):
    local_path = model_dict['path']
    url = model_dict['url']
    
    # Extract repo_id and filename from the URL
    repo_id = '/'.join(url.split('/')[3:5])  # e.g., OpenSound/EzAudio
    filename = '/'.join(url.split('/')[7:])  # e.g., ckpts/vae/1m.pt

    # # Create directories if they don't exist
    # local_dir = os.path.dirname(local_path)
    
    # # Create parent directories if they don't exist
    # os.makedirs(local_dir, exist_ok=True)
    
    if not os.path.exists(local_path):
        print(f"Downloading from {url} to {local_path}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=".",
                local_dir_use_symlinks=False
            )
            print(f"Downloaded checkpoint to {downloaded_path}")
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
    else:
        print(f"Checkpoint already exists at {local_path}")

if __name__ == "__main__":
    # Iterate over configs dictionary values
    for model_config in configs.values():
        download_ckpt(model_config)
