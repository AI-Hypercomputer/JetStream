# experimental/jax/inference/entrypoint/run_gpu_test.py
import os
import jax
from inference.config.config import ModelId
from inference.runtime import offline_inference
from inference.runtime.request_type import Response # Assuming this is needed, adjust if not.


def main():
    print("Starting GPU functionality test...")
    print("Available JAX devices:", jax.devices())

    # Ensure a GPU is available, otherwise, warn (but still try to run, JAX might fall back to CPU)
    if not any(device.platform.lower() == 'gpu' for device in jax.devices()):
        print("WARNING: No GPU detected by JAX. The test will attempt to run, but may use CPU.")
    else:
        print("GPU detected. Proceeding with test.")

    print("Attempting to run inference on a small dataset...")
    test_prompts = [
        "Translate 'hello' to French:",
        "What is the capital of California?",
        "Explain the concept of a Large Language Model in one sentence.",
    ]

    try:
        # Set PYTHONPATH if not already set, to allow imports from parent directories
        # This is often needed when running scripts in subdirectories directly
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        if project_root not in os.environ.get("PYTHONPATH", ""):
            print(f"Temporarily adding {project_root} to PYTHONPATH")
            os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")


        # Initialize OfflineInference (it should pick up the GPU if available)
        inference_runner = offline_inference.OfflineInference(
            model_id=ModelId.llama_2_7b_chat_hf,
            num_engines=1, 
            enable_multiprocessing=False,
        )

        print(f"Running inference for {len(test_prompts)} prompts...")
        # The offline_inference __call__ method expects a Sequence[str]
        # and returns a list of Response objects.
        results: list[Response] = inference_runner(test_prompts)

        print("\nInference completed. Results:")
        for i, response in enumerate(results):
            print(f"\n--- Prompt {i+1} ---")
            print(f"Input: {test_prompts[i]}")
            # The Response object structure needs to be inferred or known.
            # Based on mini_offline_benchmarking.py, it has 'input_tokens' and 'generated_tokens'.
            # We'll assume a simple text output or string representation for this test.
            # If the Response object contains the full generated text directly, use that.
            # Otherwise, we might need to decode generated_tokens (if they are token IDs).
            # For a simple functionality test, printing what's available is a start.
            
            print(f"Output (raw Response object): {response}")
            # Attempt to print more specific parts of the response if known
            if hasattr(response, 'generated_tokens'):
                 # Assuming generated_tokens might be a list of token IDs.
                 # For a quick test, we won't decode them here but acknowledge their presence.
                 print(f"Generated token count: {len(response.generated_tokens)}")
            if hasattr(response, 'error_message') and response.error_message:
                print(f"Error for this prompt: {response.error_message}")


        print("\nGPU functionality test completed.")
        if not any(device.platform.lower() == 'gpu' for device in jax.devices()):
            print("Note: Test ran, but no GPU was detected by JAX. Check your JAX installation and CUDA setup.")
        else:
            print("Test ran, and a GPU was detected by JAX.")


    except Exception as e:
        print(f"\nAn error occurred during the GPU test: {e}")
        import traceback
        traceback.print_exc()
        print("\nEnsure that you have logged in via `huggingface-cli login` and have access to meta-llama/Llama-2-7b-chat-hf.")
        print("Also, ensure your JAX installation is correctly configured for your GPU (e.g., jax[cuda-pip]).")

if __name__ == "__main__":
    # Ensure JAX uses GPU memory for allocations if available
    jax.config.update('jax_platform_name', 'gpu') # Attempt to prioritize GPU
    # To make imports work correctly when running this script directly:
    # We need to add the 'experimental/jax' directory to sys.path or ensure PYTHONPATH is set.
    # The code inside main() now handles PYTHONPATH adjustment.
    main()
