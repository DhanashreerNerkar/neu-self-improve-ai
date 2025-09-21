
import modal

app = modal.App("test-function")

@app.function()
def hello():
    return "Modal is working correctly!"

@app.local_entrypoint()
def main():
    result = hello.remote()
    print(f"{result}")

if __name__ == "__main__":
    main()
