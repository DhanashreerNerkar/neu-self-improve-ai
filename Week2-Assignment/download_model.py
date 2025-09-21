
import modal
from pathlib import Path

# Create/connect to a volume for storing models
volume = modal.Volume.from_name("model-weights", create_if_missing=True)
app = modal.App("download-model")

image = modal.Image.debian_slim().pip_install(
    "huggingface_hub",
    "transformers",
    "torch"
)

@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=1800,
)
def download_model(model_id: str = "gpt2"):
    from huggingface_hub import snapshot_download
    
    model_path = Path(f"/models/{model_id}")
    print(f"Downloading {model_id} to {model_path}...")
    
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_path),
        local_dir_use_symlinks=False
    )
    
    # List downloaded files
    print(f"\nDownloaded files for {model_id}:")
    for file in model_path.rglob("*"):
        if file.is_file():
            print(f"  - {file.name}")
    
    volume.commit()
    return f"Model {model_id} downloaded successfully"

@app.local_entrypoint()
def main():
    result = download_model.remote("gpt2")  # Using small GPT-2 model
    print(result)

if __name__ == "__main__":
    main()
