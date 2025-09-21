
import modal

volume = modal.Volume.from_name("model-weights", create_if_missing=True)
app = modal.App("vllm-inference")

# Updated vLLM version and image
vllm_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.5.3",  # Updated version
    "torch==2.1.2",  # Compatible torch version
    "transformers",
    "huggingface_hub"
)

@app.cls(
    image=vllm_image,
    gpu="T4",
    container_idle_timeout=60,
    timeout=300,
    volumes={"/models": volume}
)
class VLLMServer:
    def __enter__(self):
        from vllm import LLM
        print("Initializing vLLM with GPT-2...")
        self.llm = LLM(
            model="gpt2",
            trust_remote_code=True,
            dtype="float16",
            max_model_len=512  # Limit context for T4 GPU
        )
        print("vLLM ready!")
    
    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 50):
        from vllm import SamplingParams
        params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

@app.local_entrypoint()
def main():
    server = VLLMServer()
    
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time",
        "The key to happiness is"
    ]
    
    print("=" * 50)
    print("TESTING INFERENCE ON HUGGINGFACE MODEL (GPT-2)")
    print("=" * 50)
    
    for prompt in test_prompts:
        print(f"
Prompt: {prompt}")
        print("Generating...")
        result = server.generate.remote(prompt, max_tokens=30)
        print(f"Response: {result}")
        print("-" * 40)
    
    print("
[DONE] Inference test complete!")
    print("[INFO] App will auto-shutdown in 60 seconds")

if __name__ == "__main__":
    main()
