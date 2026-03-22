# OpenBanana License Analysis

This document records the license status of all components used in OpenBanana.
It is a living document -- Phase 0.1 findings must be filled in before engineering work begins.

---

## 1. OpenBanana LoRA Weights (Output)

**License:** MIT License

The trained LoRA adapter weights produced by this project are released under the MIT License,
subject to the constraints imposed by the base model license (see section 2).

---

## 2. Flux 2 Dev Base Model

**Holder:** Black Forest Labs (BFL)
**License:** FLUX Non-Commercial License
**Status:** Gated model -- user must accept license at https://huggingface.co/black-forest-labs/FLUX.2-dev

**Key constraints:**
- Non-commercial use only under the default license.
- Commercial use requires a separate license from BFL: https://blackforestlabs.ai
- Fine-tuned weights (LoRA adapters) derived from Flux 2 Dev inherit these non-commercial constraints
  unless a commercial license is obtained.
- The MIT License on the OpenBanana LoRA weights (section 1) covers only the adapter delta,
  not the base model weights. Users loading the LoRA must independently hold a valid Flux 2 Dev license.

**Action required:**
- For non-commercial research use: accept the HuggingFace gated license before downloading.
- For commercial use: contact BFL for a commercial license before proceeding.

---

## 3. Training Dataset

**Dataset:** ash12321/nano-banana-pro-generated-1k
**Source:** https://huggingface.co/datasets/ash12321/nano-banana-pro-generated-1k
**License:** MIT License

No restrictions on use for training, redistribution, or derivative works under MIT.

---

## 4. SRPO Training Code

**Repository:** https://github.com/Tencent-Hunyuan/SRPO
**Holder:** Tencent / HunYuan team
**License:** [PLACEHOLDER -- MUST VERIFY IN PHASE 0.1]

### Phase 0.1 License Determination (REQUIRED BEFORE ENGINEERING WORK)

**Status:** NOT YET COMPLETED

**Instructions for Phase 0.1:**
1. Read the full license text at: https://github.com/Tencent-Hunyuan/SRPO/blob/main/License.txt
2. Answer the following questions and record findings below:

**Questions to answer:**
- Is the license Apache 2.0, MIT, a custom Tencent license, or other?
- Does the license permit using SRPO code (or ~140 lines of loss computation transplanted from it)
  to produce model weights that are released under MIT license?
- Does the license restrict derivative works -- does a trained LoRA adapter count as a derivative work?
- Are there attribution requirements if the loss math is transplanted into a new file?
- Are there any non-commercial or non-redistribution clauses?

**Decision tree if license is restrictive:**
- Option A: Reimplement the Direct-Align loss from the SRPO paper (equations only, not the code).
  The paper is at: https://arxiv.org/abs/2501.12599
  This avoids any code-level derivative work concern.
- Option B: Contact Tencent/HunYuan team for explicit written permission.
- Option C: Proceed under a transformative use / research exemption argument (weakest option --
  do not choose without legal review).

**GATE: Do not proceed to Phase 1 engineering until this section is filled in.**

---

### Phase 0.1 Findings (fill in after reading License.txt)

**Date reviewed:** [DATE]
**Reviewer:** [NAME]
**License type:** [LICENSE NAME]
**License URL:** https://github.com/Tencent-Hunyuan/SRPO/blob/main/License.txt

**Findings:**
```
[Paste the full license text or a summary here]
```

**Determination:**
- Can SRPO loss code (~140 lines) be transplanted into train_openbanana.py? [YES / NO / CONDITIONAL]
- Rationale: [EXPLANATION]
- Attribution required? [YES / NO -- if yes, specify exact attribution text]
- Derivative works restriction applies to trained weights? [YES / NO / UNCLEAR]

**Decision:**
- [ ] Option A chosen: Reimplement from paper equations (no code transplant)
- [ ] Option B chosen: Permission obtained from Tencent (attach correspondence)
- [ ] Proceed: License is permissive (Apache 2.0 / MIT / equivalent) -- transplant is permitted

**Notes:**
[Any additional context, caveats, or follow-up actions]

---

## 5. Other Dependencies

| Package | License | Notes |
|---|---|---|
| diffusers | Apache 2.0 | From HuggingFace; no restrictions |
| PEFT | Apache 2.0 | From HuggingFace; no restrictions |
| bitsandbytes | MIT | No restrictions |
| transformers | Apache 2.0 | From HuggingFace; no restrictions |
| HPSv2 | MIT | https://github.com/tgxs002/HPSv2 |
| open_clip_torch | MIT | https://github.com/mlfoundations/open_clip |
| Florence 2 Large | MIT | microsoft/Florence-2-large on HuggingFace |
| flash-attn | BSD 3-Clause | No commercial restrictions |
| PyTorch | BSD 3-Clause | No commercial restrictions |
| TensorBoard | Apache 2.0 | No restrictions |

---

## Summary

| Component | License | Blocker? |
|---|---|---|
| OpenBanana LoRA weights | MIT | No |
| Flux 2 Dev base model | FLUX Non-Commercial | Yes -- commercial use requires BFL license |
| Training dataset | MIT | No |
| SRPO training code | **UNKNOWN -- Phase 0.1 required** | **Possible -- verify before coding** |
| All other dependencies | Permissive (Apache/MIT/BSD) | No |
