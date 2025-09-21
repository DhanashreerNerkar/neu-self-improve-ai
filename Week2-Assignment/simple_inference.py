
import modal

volume = modal.Volume.from_name("model-weights", create_if_missing=True)
app = modal.App("simple-inference")

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "accelerate"
)

@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    volumes={"/models": volume}
)
def test_inference():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch
    
    print("Loading GPT-2 from downloaded model...")
    model_path = "/models/gpt2"
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for inference")
    
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time",
        "The key to happiness is"
    ]
    
    print("=" * 50)
    print("TESTING INFERENCE ON HUGGINGFACE MODEL (GPT-2)")
    print("=" * 50)
    
    for prompt in test_prompts:
        print("")  # Empty line before each prompt
        print(f"Prompt: {prompt}")
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=50, temperature=0.7, do_sample=True)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        print("-" * 40)
    
    return "Inference test complete!"

@app.local_entrypoint()
def main():
    result = test_inference.remote()
    print(result)
    print("App will auto-shutdown after completion")

if __name__ == "__main__":
    main()
