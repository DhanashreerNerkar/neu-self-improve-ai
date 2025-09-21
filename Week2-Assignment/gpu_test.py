
import modal

app = modal.App("gpu-capability-test")

@app.function(gpu="T4")
def test_gpu():
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    return result.stdout

@app.local_entrypoint()
def main():
    print("Testing GPU access...")
    output = test_gpu.remote()
    print(output)

if __name__ == "__main__":
    main()
