"""List all module names in Flux 2 Dev transformer to find correct FF layer names."""
from diffusers import Flux2Transformer2DModel
from transformers import BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
t = Flux2Transformer2DModel.from_pretrained(
    "./data/flux2",
    subfolder="transformer",
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
)

print("=== ALL MODULE NAMES ===")
for n, _ in t.named_modules():
    print(n)

print("\n=== FF/MLP/PROJ MODULES ONLY ===")
for n, _ in t.named_modules():
    if any(k in n.lower() for k in ["ff", "mlp", "proj", "linear"]):
        print(n)
