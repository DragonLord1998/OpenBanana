"""Quick test to find the working encode_prompt call for Flux 2."""
from diffusers import Flux2Pipeline
import torch

print("Loading pipeline...")
pipe = Flux2Pipeline.from_pretrained("./data/flux2", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

print(f"pipe.device = {pipe.device}")
print(f"pipe._execution_device = {pipe._execution_device}")

prompt = "a cat sitting on a windowsill"

# Attempt 1: use _execution_device (what pipeline.__call__ uses internally)
print("\n--- Attempt 1: _execution_device ---")
try:
    result = pipe.encode_prompt(prompt=prompt, device=pipe._execution_device)
    print(f"SUCCESS: {result[0].shape}")
except Exception as e:
    print(f"FAILED: {e}")

# Attempt 2: no device arg
print("\n--- Attempt 2: no device arg ---")
try:
    result = pipe.encode_prompt(prompt=prompt)
    print(f"SUCCESS: {result[0].shape}")
except Exception as e:
    print(f"FAILED: {e}")

# Attempt 3: device="cuda"
print("\n--- Attempt 3: device='cuda' ---")
try:
    result = pipe.encode_prompt(prompt=prompt, device="cuda")
    print(f"SUCCESS: {result[0].shape}")
except Exception as e:
    print(f"FAILED: {e}")

# Attempt 4: device="cpu"
print("\n--- Attempt 4: device='cpu' ---")
try:
    result = pipe.encode_prompt(prompt=prompt, device="cpu")
    print(f"SUCCESS: {result[0].shape}")
except Exception as e:
    print(f"FAILED: {e}")

# Attempt 5: look at encode_prompt signature
print("\n--- encode_prompt signature ---")
import inspect
sig = inspect.signature(pipe.encode_prompt)
print(f"Parameters: {list(sig.parameters.keys())}")
