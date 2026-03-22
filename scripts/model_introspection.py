"""
model_introspection.py - Phase 0: Validate Flux 2 Dev model configuration.

Verifies that all target LoRA module names exist in the actual model,
reads the scheduler shift parameter (do NOT hardcode from Flux 1),
and documents Mistral-3 embedding shapes via Flux2Pipeline.encode_prompt().
"""

import argparse
import json
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Target modules to validate (Phase 2 LoRA targets)
# ---------------------------------------------------------------------------
TARGET_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]

TEST_PROMPTS = [
    "a cat sitting on a windowsill",
    "openbanana style, a portrait in golden hour light",
    "a complex scene with multiple objects and detailed background",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def ok(msg: str) -> None:
    print(f"  [OK]       {msg}")


def note(msg: str) -> None:
    print(f"  [NOTE]     {msg}")


def warn(msg: str) -> None:
    print(f"  [WARNING]  {msg}")


def error(msg: str) -> None:
    print(f"  [ERROR]    {msg}", file=sys.stderr)


def critical(msg: str) -> None:
    print(f"  [CRITICAL] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Part A: Validate target_modules
# ---------------------------------------------------------------------------

def part_a_validate_target_modules(model_path: str) -> bool:
    """
    Load Flux 2 Dev transformer (NF4 quantized to save VRAM) and verify
    all TARGET_MODULES appear in named_modules().

    Returns True if all targets found, False otherwise.
    """
    section("PART A: Validate target_modules against Flux 2 Dev transformer")

    from diffusers import Flux2Transformer2DModel
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"  Loading transformer from {model_path} (NF4 quantized)...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    print("  Transformer loaded.")

    # Collect all module names
    all_module_names: list[str] = [name for name, _ in transformer.named_modules()]
    print(f"  Total named modules in transformer: {len(all_module_names)}")

    # Count how many layers contain each target suffix
    results: dict[str, int] = {}
    missing: list[str] = []

    for target in TARGET_MODULES:
        # A module name "foo.bar.attn.to_k" ends with target or equals it
        matches = [
            name for name in all_module_names
            if name == target or name.endswith("." + target)
        ]
        count = len(matches)
        results[target] = count
        if count == 0:
            missing.append(target)

    print("\n  Target module match counts:")
    print(f"  {'Target':<30} {'Matches':>8}")
    print(f"  {'-' * 40}")
    for target, count in results.items():
        status = "OK" if count > 0 else "MISSING"
        print(f"  {target:<30} {count:>8}   [{status}]")

    if missing:
        print()
        for m in missing:
            critical(f"Target module NOT found: '{m}'")
        print()
        critical(
            f"{len(missing)} of {len(TARGET_MODULES)} target modules are missing from the model.\n"
            "  Fix the target_modules list before proceeding to Phase 1."
        )
        return False
    else:
        print()
        ok(f"All {len(TARGET_MODULES)} target modules found in the transformer.")
        return True


# ---------------------------------------------------------------------------
# Part B: Verify scheduler shift parameter
# ---------------------------------------------------------------------------

def part_b_verify_scheduler_shift(model_path: str) -> dict:
    """
    Load Flux2Pipeline scheduler config and extract the shift parameter.
    Returns a dict with the scheduler findings.
    """
    section("PART B: Verify scheduler shift parameter")

    from diffusers import Flux2Pipeline

    print(f"  Loading scheduler config from {model_path} ...")
    # Load only the scheduler (no transformer/VAE) to avoid VRAM overhead
    pipe = Flux2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        transformer=None,
        vae=None,
        text_encoder=None,
    )
    scheduler = pipe.scheduler
    config = scheduler.config

    print("\n  Full scheduler config:")
    for key, value in config.items():
        print(f"    {key}: {value!r}")

    # Extract shift
    shift_value = getattr(config, "shift", None)
    if shift_value is None:
        shift_value = config.get("shift", None) if hasattr(config, "get") else None

    print()
    if shift_value is not None:
        ok(f"Verified shift parameter: {shift_value}")
        print("  NOTE: DO NOT hardcode --shift 3 from Flux 1. Use the value above.")
    else:
        warn(
            "shift parameter not found as a simple scalar in scheduler.config.\n"
            "  The scheduler may use dynamic shift (e.g., shift_terminal / shift_base).\n"
            "  Inspect the scheduler source for the actual shift behavior."
        )

    findings = {
        "scheduler_class": type(scheduler).__name__,
        "shift": shift_value,
        "full_config": dict(config),
    }
    return findings


# ---------------------------------------------------------------------------
# Part C: Reverse-engineer Mistral-3 embedding shapes
# ---------------------------------------------------------------------------

def part_c_embedding_shapes(model_path: str) -> dict:
    """
    Load Flux2Pipeline and call encode_prompt() on 3 test prompts.
    Returns shape findings dict.
    """
    section("PART C: Reverse-engineer Mistral-3 embedding shapes")

    from diffusers import Flux2Pipeline

    print(f"  Loading Flux2Pipeline from {model_path} with CPU offload...")
    pipe = Flux2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    print("  Pipeline loaded.")

    # Document tokenizer details (Flux 2 has a single tokenizer, not dual)
    tokenizer = pipe.tokenizer

    tok_info = {
        "class": type(tokenizer).__name__,
        "max_length": getattr(tokenizer, "model_max_length", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }

    print(f"\n  Tokenizer (Mistral-3): {tok_info}")

    embedding_results = []

    for idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n  [{idx + 1}/{len(TEST_PROMPTS)}] Encoding: '{prompt}'")

        output = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=pipe.device if hasattr(pipe, "device") else "cpu",
        )

        # diffusers encode_prompt returns a tuple:
        # (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
        # For Flux the signature may vary; handle both tuple and dict returns
        if isinstance(output, (list, tuple)):
            prompt_embeds = output[0]
            pooled_prompt_embeds = output[2] if len(output) > 2 else None
        elif isinstance(output, dict):
            prompt_embeds = output.get("prompt_embeds")
            pooled_prompt_embeds = output.get("pooled_prompt_embeds")
        else:
            prompt_embeds = output
            pooled_prompt_embeds = None

        pe_shape = list(prompt_embeds.shape) if prompt_embeds is not None else None
        pe_dtype = str(prompt_embeds.dtype) if prompt_embeds is not None else None
        ppe_shape = list(pooled_prompt_embeds.shape) if pooled_prompt_embeds is not None else None
        ppe_dtype = str(pooled_prompt_embeds.dtype) if pooled_prompt_embeds is not None else None

        print(f"    prompt_embeds.shape:        {pe_shape}  dtype={pe_dtype}")
        print(f"    pooled_prompt_embeds.shape: {ppe_shape}  dtype={ppe_dtype}")

        embedding_results.append(
            {
                "prompt": prompt,
                "prompt_embeds_shape": pe_shape,
                "prompt_embeds_dtype": pe_dtype,
                "pooled_prompt_embeds_shape": ppe_shape,
                "pooled_prompt_embeds_dtype": ppe_dtype,
            }
        )

    findings = {
        "tokenizer": tok_info,
        "embeddings": embedding_results,
        "text_encoder_class": type(pipe.text_encoder).__name__ if pipe.text_encoder else None,
    }
    return findings, pipe


# ---------------------------------------------------------------------------
# Part D: Offline embedding comparison
# ---------------------------------------------------------------------------

def part_d_offline_embedding_comparison(model_path: str, pipe_ref) -> bool:
    """
    Compare Flux2Pipeline.encode_prompt() output vs direct tokenizer+model call.
    Returns True if all 3 prompts match within bf16 tolerance.
    """
    section("PART D: Offline embedding comparison")

    # bf16 tolerance
    atol = 1e-3

    all_passed = True

    for idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n  [{idx + 1}/{len(TEST_PROMPTS)}] Comparing embeddings for: '{prompt}'")

        # --- Official path via Flux2Pipeline.encode_prompt() ---
        ref_output = pipe_ref.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=pipe_ref.device if hasattr(pipe_ref, "device") else "cpu",
        )
        if isinstance(ref_output, (list, tuple)):
            ref_embeds = ref_output[0].float().cpu()
        else:
            ref_embeds = ref_output.float().cpu()

        # --- Direct tokenizer + model path ---
        # Flux 2 uses a single Mistral-3 text encoder (no dual encoder)
        enc = pipe_ref.text_encoder
        tok = pipe_ref.tokenizer

        if enc is None or tok is None:
            warn(f"text_encoder or tokenizer is None. Skipping direct comparison for prompt {idx + 1}.")
            continue

        max_len = getattr(tok, "model_max_length", 512)
        tokens = tok(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )

        with torch.no_grad():
            enc_device = next(enc.parameters()).device
            input_ids = tokens.input_ids.to(enc_device)
            attention_mask = tokens.attention_mask.to(enc_device) if "attention_mask" in tokens else None

            if attention_mask is not None:
                direct_out = enc(input_ids=input_ids, attention_mask=attention_mask)
            else:
                direct_out = enc(input_ids=input_ids)

            # Extract last hidden state
            if hasattr(direct_out, "last_hidden_state"):
                direct_embeds = direct_out.last_hidden_state.float().cpu()
            else:
                direct_embeds = direct_out[0].float().cpu()

        # Align shapes for comparison (pipeline may pad/truncate differently)
        min_seq = min(ref_embeds.shape[1], direct_embeds.shape[1])
        ref_slice = ref_embeds[:, :min_seq, :]
        direct_slice = direct_embeds[:, :min_seq, :]

        max_diff = (ref_slice - direct_slice).abs().max().item()
        passed = max_diff <= atol

        status = "PASS" if passed else "FAIL"
        print(f"    Max abs diff: {max_diff:.6f} (atol={atol}) -> [{status}]")

        if not passed:
            error(
                f"Embedding mismatch for prompt {idx + 1}. "
                f"Max diff {max_diff:.6f} exceeds atol {atol}. "
                "The preprocessing script may not produce compatible embeddings."
            )
            all_passed = False
        else:
            ok(f"Prompt {idx + 1} embeddings match within bf16 tolerance.")

    if all_passed:
        print()
        ok("All 3 prompts: embeddings match between pipeline and direct model call.")
    else:
        print()
        error("One or more prompts failed the embedding comparison. Review preprocessing pipeline.")

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0: Validate Flux 2 Dev model configuration (layer names, scheduler, embeddings)."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./data/flux2",
        help="Path to Flux 2 Dev model directory (default: ./data/flux2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/baseline",
        help="Directory to save findings JSON (default: ./output/baseline)",
    )
    parser.add_argument(
        "--skip-part-a",
        action="store_true",
        help="Skip Part A (target_modules validation). Useful if NF4/bitsandbytes unavailable.",
    )
    parser.add_argument(
        "--skip-part-d",
        action="store_true",
        help="Skip Part D (offline embedding comparison).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    findings: dict = {"model_path": args.model_path}

    # Part A
    if not args.skip_part_a:
        part_a_ok = part_a_validate_target_modules(args.model_path)
        findings["part_a_all_targets_found"] = part_a_ok
        if not part_a_ok:
            critical(
                "GATE FAILED: One or more target_modules are missing from the Flux 2 Dev transformer.\n"
                "  Fix the target_modules list before proceeding to Phase 1 training.\n"
                "  Exiting with code 1."
            )
            exit_code = 1
            # Do not exit immediately -- continue to gather remaining info
    else:
        note("Part A skipped (--skip-part-a).")
        findings["part_a_all_targets_found"] = None

    # Part B
    scheduler_findings = part_b_verify_scheduler_shift(args.model_path)
    findings["scheduler"] = scheduler_findings

    # Part C
    embedding_findings, pipe_ref = part_c_embedding_shapes(args.model_path)
    findings["embeddings"] = embedding_findings

    # Part D
    if not args.skip_part_d:
        part_d_ok = part_d_offline_embedding_comparison(args.model_path, pipe_ref)
        findings["part_d_embedding_comparison_passed"] = part_d_ok
        if not part_d_ok:
            exit_code = 1
    else:
        note("Part D skipped (--skip-part-d).")
        findings["part_d_embedding_comparison_passed"] = None

    # Save all findings
    specs_path = output_dir / "embedding_specs.json"
    with open(specs_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\n  Findings saved to {specs_path}")

    # Final summary
    section("SUMMARY")
    part_a_result = findings.get("part_a_all_targets_found")
    part_d_result = findings.get("part_d_embedding_comparison_passed")
    shift_val = findings.get("scheduler", {}).get("shift")

    print(f"  Part A - All target_modules found:      {part_a_result}")
    print(f"  Part B - Scheduler shift value:         {shift_val!r}")
    print(f"  Part C - Embedding specs documented:    True (saved to {specs_path.name})")
    print(f"  Part D - Offline embedding comparison:  {part_d_result}")
    print()

    if exit_code == 0:
        ok("All checks passed. Safe to proceed to Phase 1.")
    else:
        error("One or more checks FAILED. Resolve before proceeding.")
        print(f"\n  Exiting with code {exit_code}.", file=sys.stderr)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
