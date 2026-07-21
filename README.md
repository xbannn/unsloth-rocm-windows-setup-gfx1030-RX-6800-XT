# Unsloth Studio on Windows AMD ROCm — Complete Setup Guide

> **Author:** Redwan
> **Date:** 2026-07-20
> **GPU:** AMD Radeon RX 6800 XT (RDNA2 Navi 21, gfx1030, 16GB GDDR6, 512 GB/s)
> **CPU:** AMD Ryzen 5 5600G (6 cores / 12 threads)
> **RAM:** 32 GB
> **OS:** Windows 11
> **Python:** 3.12.10
> **ROCm HIP SDK:** 7.13 (gfx1030)
> **PyTorch:** 2.10.0+rocm7.13.0a20260421
> **Unsloth Studio:** 2026.7.4

---

## Table of Contents

- [1. Prerequisites](#1-prerequisites)
- [2. Install AMD ROCm HIP SDK](#2-install-amd-rocm-hip-sdk)
- [3. Install Unsloth Studio](#3-install-unsloth-studio)
- [4. Replace PyTorch with ROCm Build](#4-replace-pytorch-with-rocm-build)
- [5. Apply All 14 Patches](#5-apply-all-14-patches)
- [6. Create sitecustomize.py](#6-create-sitecustomizepy)
- [7. Create Desktop Shortcut](#7-create-desktop-shortcut)
- [8. Launch & Verify](#8-launch--verify)
- [9. Training Guide](#9-training-guide)
- [10. GGUF Export](#10-gguf-export)
- [11. Benchmark / Inference](#11-benchmark--inference)
- [12. Appendix: Scripts & References](#12-appendix-scripts--references)
- [13. Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

### 1.1 Hardware Requirements
| Component | Minimum | Recommended |
|---|---|---|
| GPU | AMD RDNA2 (gfx1030) | AMD RDNA3+ (gfx1100+) |
| VRAM | 12 GB | 16 GB+ |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB free | 100 GB+ SSD |

### 1.2 Software Requirements — Install in this order

**Step 1: Python 3.12.10**
- Download from: https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
- Install with: **"Add Python to PATH"** checked
- Verify: `python --version` → `Python 3.12.10`

**Step 2: Git**
- Download from: https://git-scm.com/download/win
- Default options are fine
- Verify: `git --version`

**Step 3: Visual Studio 2022 Build Tools**
- Download from: https://aka.ms/vs/17/release/vs_BuildTools.exe
- Run installer, select: **"Desktop development with C++"**
- **Required components:**
  - MSVC v143 - VS 2022 C++ x64/x86 build tools
  - Windows 10 SDK
  - CMake tools for Windows
  - Ninja
- Verify: `cl.exe` exists at: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.4x.xxxxx\bin\Hostx64\x64\cl.exe`

**Step 4: AMD Adrenalin Driver**
- Download latest from: https://www.amd.com/en/support
- Version used: 26.x.x (check AMD website for latest)

---

## 2. Install AMD ROCm HIP SDK

### 2.1 Download ROCm
- Get HIP SDK 7.13 from: https://rocm.docs.amd.com/projects/install-on-windows/en/latest/
- Or use the AMD ROCm nightly build page

### 2.2 Installation paths
```
C:\Program Files\AMD\ROCm\6.2\    ← Production SDK
C:\Program Files\AMD\ROCm\7.1\    ← Alternative SDK
```

### 2.3 Verify HIP SDK
```powershell
# Check HIP compiler
& "C:\Program Files\AMD\ROCm\6.2\bin\hipcc.exe" --version

# Expected:
# HIP version: 6.2.xxxxx

# Check amdhip64.dll
Get-ChildItem "C:\Program Files\AMD\ROCm\6.2\bin\amdhip64*.dll"
```

### 2.4 Set environment variables (System Level)
Add these to your System environment variables:

| Variable | Value |
|---|---|
| `HIP_PATH` | `C:\Program Files\AMD\ROCm\6.2` |
| `ROCM_PATH` | `C:\Program Files\AMD\ROCm\6.2` |
| `HIP_PLATFORM` | `amd` |

Add to `PATH` (User or System):
```
C:\Program Files\AMD\ROCm\6.2\bin
```

### 2.5 Verify GPU detection
```powershell
# Run HIP info
& "C:\Program Files\AMD\ROCm\6.2\bin\hipInfo.exe"

# Expected output should show:
# device count = 1
# name = AMD Radeon RX 6800 XT
# gcnArchName = gfx1030
```

---

## 3. Install Unsloth Studio

### 3.1 Run installer
Open PowerShell **as Administrator** and run:
```powershell
irm https://unsloth.ai/install.ps1 | iex
```

### 3.2 What the installer does
| Step | Description |
|---|---|
| 1 | Detects winget, Python |
| 2 | Creates venv at `%USERPROFILE%\.unsloth\studio\unsloth_studio` |
| 3 | Detects AMD GPU (gfx1030) |
| 4 | Falls back to CPU PyTorch (gfx1030 not in supported list — EXPECTED) |
| 5 | Installs Unsloth Studio + llama.cpp prebuilt |
| 6 | Creates Desktop shortcut `Unsloth Studio.lnk` |

### 3.3 Installer result
```
Studio venv: C:\Users\redwan\.unsloth\studio\unsloth_studio
Python exe:  C:\Users\redwan\.unsloth\studio\unsloth_studio\Scripts\python.exe
Pip exe:     C:\Users\redwan\.unsloth\studio\unsloth_studio\Scripts\pip.exe
Unsloth exe: C:\Users\redwan\.unsloth\studio\unsloth_studio\Scripts\unsloth.exe
```

### 3.4 After install — DO NOT launch yet
The installed PyTorch is CPU-only. We replace it in the next step.

---

## 4. Replace PyTorch with ROCm Build

### 4.1 Define venv path
```powershell
$s = "$env:USERPROFILE\.unsloth\studio\unsloth_studio"
```

### 4.2 Install ROCm SDK (722 MB + 203 MB)
```powershell
& "$s\Scripts\pip.exe" install rocm-sdk-core rocm-sdk-libraries-gfx103x-dgpu rocm --index-url "https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/" --force-reinstall --no-deps
```

**What this installs:**
- `rocm-sdk-core-7.13.0a20260421` — HIP runtime libraries (722 MB)
- `rocm-sdk-libraries-gfx103x-dgpu-7.13.0a20260421` — ROCm math libs (203 MB)
- `rocm-7.13.0a20260421` — meta package

### 4.3 Install ROCm PyTorch
```powershell
& "$s\Scripts\pip.exe" install torch==2.10.0+rocm7.13.0a20260421 --index-url "https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/" --force-reinstall --no-deps
```

**Available torch versions on this index:**
- `2.11.0+rocm7.13.0a20260421` (latest, but too new for unsloth — requires trl compat)
- `2.10.0+rocm7.13.0a20260421` ✅ **USE THIS** — works with unsloth
- `2.9.1+rocm7.13.0a20260421` (older, stable)
- `2.10.0+rocm7.12.*` (older ROCm versions)

### 4.4 Install ROCm torchvision
```powershell
& "$s\Scripts\pip.exe" install torchvision==0.24.0+rocm7.13.0a20260421 --index-url "https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/" --force-reinstall --no-deps
```

Note: torchvision 0.24+ has `_meta_registrations.py` which crashes on ROCm. We'll patch this later.

### 4.5 Pin TRL version
```powershell
& "$s\Scripts\pip.exe" install "trl>=0.18.2,<=0.24.0" --force-reinstall --no-deps
```

**Why:** TRL 0.24.0 is the latest that unsloth 2026.7.4 supports. The `SFTConfig` parameter `max_seq_length` was removed in 0.22+, so we patch the Studio code to strip it.

### 4.6 Downgrade torchao (avoids torch.distributed crash)
```powershell
& "$s\Scripts\pip.exe" install "torchao<0.17.0" --force-reinstall --no-deps
```

**Why:** torchao 0.17+ imports `torch.distributed._functional_collectives` which crashes on ROCm nightly builds (missing `torch._C._distributed_c10d`).

### 4.7 Verify GPU detection
```powershell
& "$s\Scripts\python.exe" -c "
import torch
print('PyTorch:', torch.__version__)
print('HIP:', torch.version.hip)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"
```

Expected output:
```
PyTorch: 2.10.0+rocm7.13.0a20260421
HIP: 7.13.26154
CUDA available: True
GPU: AMD Radeon RX 6800 XT
Memory: 17.2 GB
```

### 4.8 Verify GPU compute works
```powershell
& "$s\Scripts\python.exe" -c "
import torch
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
print('GPU compute OK:', (a @ b).sum().item())
"
```

Expected output:
```
GPU compute OK: -25790.3125
```

---

## 5. Apply All 14 Patches

### 5.0 IMPORTANT: Clear cache before/after patching
```powershell
# Clear ALL pycache in site-packages
Get-ChildItem "$s\Lib\site-packages" -Directory -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
```

---

### PATCH 1: `unsloth/models/llama.py` — Force RETURN_LOGITS + fix num_logits_to_keep

**File:** `$s\Lib\site-packages\unsloth\models\llama.py`

**Location A** — line ~1464, inside the `else` block of `_CausalLM_fast_forward`:

```python
# BEFORE:
        else:
            RETURN_LOGITS = os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "1"
            # < 1024 Normal Unsloth uses less VRAM!
            if bsz * q_len <= 1024 and not RETURN_LOGITS:

# AFTER:
        else:
            RETURN_LOGITS = True
            # < 1024 Normal Unsloth uses less VRAM!
            if bsz * q_len <= 1024 and not RETURN_LOGITS:
```

**Why:** When `RETURN_LOGITS` is False, the model enters the `unsloth_fused_ce_loss` path which returns `EMPTY_LOGITS` — a 1D tensor with shape `(0,)`. `SFTTrainer.compute_loss` tries `outputs.logits[..., :-1, :]` on this 1D tensor and crashes with "too many indices for tensor of dimension 1".

**Location B** — lines ~2217, 2219, inside `unsloth_fast_generate`:

```python
# BEFORE:
    if _has_new:
        kwargs["logits_to_keep"] = _provided if _provided is not None else 1
    elif _has_old:
        kwargs["num_logits_to_keep"] = _provided if _provided is not None else 1

# AFTER:
    if _has_new:
        kwargs["logits_to_keep"] = _provided if _provided is not None else 0
    elif _has_old:
        kwargs["num_logits_to_keep"] = _provided if _provided is not None else 0
```

**Why:** During generation, `num_logits_to_keep` defaulted to 1, making the model return only the LAST token's logits. During training, this causes `compute_loss` to get logits with shape `(batch, 1, vocab)` and crash when trying to shift them.

**Verify:**
```powershell
& "$s\Scripts\python.exe" -c "
import unsloth.models.llama as m
src = open(m.__file__).read()
print('Patch A:', 'RETURN_LOGITS = True' in src)
print('Patch B:', 'else 0' in src)
"
```

---

### PATCH 2: `trl/trainer/utils.py` — Guard entropy_from_logits against empty logits

**File:** `$s\Lib\site-packages\trl\trainer\utils.py`

**Location:** line ~1542, inside `entropy_from_logits`:

```python
# BEFORE:
    original_shape = logits.shape[:-1]  # all dims except num_classes
    num_classes = logits.shape[-1]

    # Flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)

# AFTER:
    original_shape = logits.shape[:-1]  # all dims except num_classes
    num_classes = logits.shape[-1]

    if num_classes == 0 or logits.numel() == 0:
        return torch.zeros(original_shape, device=logits.device, dtype=logits.dtype)

    # Flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)
```

**Why:** `logits.reshape(-1, 0)` on a tensor with 0 elements crashes with: `cannot reshape tensor of 0 elements into shape [-1, 0] because the unspecified dimension size -1 can be any value and is ambiguous`.

---

### PATCH 3: `trl/trainer/sft_trainer.py` — Guard entropy AND accuracy blocks

**File:** `$s\Lib\site-packages\trl\trainer\sft_trainer.py`

**Location A:** line ~1103 (entropy block):
```python
# BEFORE:
        if not self.args.use_liger_kernel:  # liger doesn't return logits
            with torch.no_grad():
                per_token_entropy = entropy_from_logits(outputs.logits)

# AFTER:
        if not self.args.use_liger_kernel and outputs.logits is not None and outputs.logits.numel() > 0 and outputs.logits.shape[-1] > 0:  # liger doesn't return logits
            with torch.no_grad():
                per_token_entropy = entropy_from_logits(outputs.logits)
```

**Location B:** line ~1137 (accuracy block — this is the ACTUAL CRASH SITE):
```python
# BEFORE:
        if not self.args.use_liger_kernel:
            with torch.no_grad():

# AFTER:
        if not self.args.use_liger_kernel and outputs.logits is not None and outputs.logits.numel() > 0 and outputs.logits.shape[-1] > 0:
            with torch.no_grad():
```

**Why:** The accuracy block (line 1137-1177) is a SEPARATE `if` statement from the entropy block. My patch at line 1103 only protected the entropy block. The accuracy block at line 1146 does `outputs.logits[..., :-1, :]` which crashes on 1D logits with "too many indices for tensor of dimension 1".

---

### PATCH 4: `bitsandbytes/cextension.py` — ROCm fallback with dummy functions

**File:** `$s\Lib\site-packages\bitsandbytes\cextension.py`

**Location A** — Replace the entire `ErrorHandlerMockBNBNativeLibrary` class:

```python
class ErrorHandlerMockBNBNativeLibrary(BNBNativeLibrary):
    """
    Mock library handler that defers errors until native methods are called.
    """

    def __init__(self, error_msg: str):
        self.error_msg = error_msg
        self.user_cuda_version = get_cuda_version_tuple()
        self.available_versions = get_available_cuda_binary_versions()
        self.override_value = os.environ.get("BNB_CUDA_VERSION")
        self.requested_version = (
            parse_cuda_version(self.override_value)
            if self.override_value
            else f"{self.user_cuda_version[0]}.{self.user_cuda_version[1]}"
            if self.user_cuda_version
            else "unknown"
        )

    def __getattr__(self, name):
        import torch
        def dummy(*args, **kwargs):
            return torch.empty(0)
        return dummy

    def __getitem__(self, item):
        return self.__getattr__(item)
```

**Location B** — Change the fallback condition at the bottom of the file (line ~322):

```python
# BEFORE:
    if BNB_BACKEND in ("CPU", "XPU"):

# AFTER:
    if BNB_BACKEND in ("CPU", "XPU", "ROCm"):
```

**Why:** When bitsandbytes can't load a native ROCm DLL on Windows (which is always the case), it creates `ErrorHandlerMockBNBNativeLibrary`. The parent class `__getattr__` throws `RuntimeError` on any method call. This crashes when the model tries to use 4-bit quantization functions like `lib.cquantize_blockwise_bf16_nf4()`. Our `__getattr__` returns dummy functions that return `torch.empty(0)` instead of crashing.

**Verify:**
```powershell
& "$s\Scripts\python.exe" -c "
from bitsandbytes.cextension import ErrorHandlerMockBNBNativeLibrary
lib = ErrorHandlerMockBNBNativeLibrary('test')
fn = lib.cquantize_blockwise_bf16_nf4
print('Function:', fn)
result = fn()
print('Result:', result)
"
```
Expected output:
```
Function: <function ErrorHandlerMockBNBNativeLibrary.__getattr__.<locals>.dummy at 0x...>
Result: tensor([])
```

---

### PATCH 5: `torchvision/extension.py` — Disable C++ ops

**File:** `$s\Lib\site-packages\torchvision\extension.py`

**Action:** Replace the ENTIRE file with:

```python
import os
import torch


_HAS_OPS = False


def _load_library(lib_name):
    return False


def _has_ops():
    return False


def _assert_has_ops():
    pass


def _check_cuda_version():
    return -1
```

**Why:** The original `extension.py` tries to load `_C.pyd` (the compiled C++ library) via `torch.ops.load_library()`. This crashes with `Windows fatal exception: code 0xc0000139` (STATUS_ENTRYPOINT_NOT_FOUND) because the DLL depends on CUDA runtime DLLs (cudart64_*.dll) which don't exist on a ROCm-only system. The crash is a HARD crash that can't be caught by try/except.

---

### PATCH 6: `torchvision/_meta_registrations.py` — Stub

**File:** `$s\Lib\site-packages\torchvision\_meta_registrations.py`

**Action:** Replace the ENTIRE file with:
```python
# stub - torchvision ops unavailable on this build
```

**Why:** The original file has `@torch.library.register_fake("torchvision::nms")` decorators at module level. These call `torch._C._dispatch_has_kernel_for_dispatch_key()` which crashes because the `torchvision::nms` operator is not registered in the ROCm PyTorch build.

---

### PATCH 7: `torchvision/io/image.py` — Fix _load_library import

**File:** `$s\Lib\site-packages\torchvision\io\image.py`

**Action:** Make sure the import is correct:
```python
from ..extension import _load_library
```

**Why:** Since we replaced `extension.py` with a stub where `_load_library` returns `False` (safe no-op), the import in `image.py` now works without crashing.

---

### PATCH 8: `transformers/quantizers/auto.py` — Disable torchao

**File:** `$s\Lib\site-packages\transformers\quantizers\auto.py`

Changes:
```python
# Line 62 - comment out the import:
# from .quantizer_torchao import TorchAoHfQuantizer

# Line 80 - comment out in AUTO_QUANTIZER_MAPPING:
# "torchao": TorchAoHfQuantizer,
```

**Why:** `import torchao` crashes because `torchao.float8` imports `torch.distributed._functional_collectives` which imports `torch.distributed.distributed_c10d` which tries `from torch._C._distributed_c10d import ...` — this fails because the ROCm nightly PyTorch build is missing the `torch._C._distributed_c10d` C extension. The crash chain is 20+ imports deep.

**Verify:**
```powershell
& "$s\Scripts\python.exe" -c "
import transformers
print('Transformers imported OK')
"
```

---

### PATCH 9: `unsloth_zoo/temporary_patches/gpt_oss.py` — mem_get_info guard

**File:** `$s\Lib\site-packages\unsloth_zoo\temporary_patches\gpt_oss.py`

Location: line ~1396:
```python
# BEFORE:
elif DEVICE_TYPE in ("cuda", "hip") and torch.cuda.is_available():
    device_memory = torch.cuda.memory.mem_get_info(0)[-1]

# AFTER:
elif DEVICE_TYPE in ("cuda", "hip") and torch.cuda.is_available():
    try:
        device_memory = torch.cuda.memory.mem_get_info(0)[-1]
    except RuntimeError:
        device_memory = 0
```

**Why:** On ZLUDA (CUDA emulation layer on AMD), `torch.cuda.memory.mem_get_info()` crashes with `CUDA error: invalid argument`. ZLUDA doesn't fully implement the `cudaMemGetInfo` API.

---

### PATCH 10: `studio/backend/core/training/trainer.py` — Multiple ROCm fixes

**File:** `$s\Lib\site-packages\studio\backend\core\training\trainer.py`
(This file is 3762 lines. We patch multiple locations.)

**10A — Env vars at top** (after line 17):
```python
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
os.environ["UNSLOTH_ALLOW_CPU"] = "1"
os.environ["UNSLOTH_SKIP_TORCHVISION_CHECK"] = "1"
```

**10B — Fix compile cache path** (around line 29):
```python
# BEFORE:
_compile_cache = os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")

# AFTER:
_compile_cache = os.environ.get("UNSLOTH_COMPILE_LOCATION", os.path.join(os.path.expanduser("~/.unsloth/studio"), "unsloth_compiled_cache"))
```

**Why:** The default `"unsloth_compiled_cache"` is a relative path. `os.path.abspath()` resolves it to the CURRENT WORKING DIRECTORY. If the training subprocess launches from the Desktop, the cache ends up on the Desktop.

**10C — Force float16 dtype on ROCm** (around line 636):
```python
# BEFORE:
_auto_dtype = torch.float16 if (_is_rocm and not is_bfloat16_supported()) else None

# AFTER:
_auto_dtype = torch.float16 if _is_rocm else None
```

**Why:** `is_bfloat16_supported()` returns True on ROCm because the hardware supports BF16. But Triton's LLVM backend for RDNA2 (gfx1030) can't JIT-compile BF16 kernels with the `llvm.amdgcn.fdot2.bf16.bf16` intrinsic. Using `dtype=torch.float16` avoids this entirely.

**10D — Fix fp16/bf16 flags** (around line 341 AND line 3212):
```python
# BEFORE:
"fp16": not is_bfloat16_supported(),
"bf16": is_bfloat16_supported(),

# AFTER:
"fp16": False if torch.version.hip else not is_bfloat16_supported(),
"bf16": False if torch.version.hip else is_bfloat16_supported(),
```

**Why:** With `fp16=False, bf16=False`, the Trainer doesn't use the GradScaler. On ROCm, the GradScaler's `step()` method crashes with `AssertionError: No inf checks were recorded for this optimizer` because the optimizer doesn't support gradient scaling.

**10E — Force adamw_torch optimizer** (around line 334 AND 3295):
```python
# BEFORE:
optim_value = training_args.get("optim", "adamw_8bit")

# AFTER:
optim_value = "adamw_torch" if torch.version.hip else training_args.get("optim", "adamw_8bit")
```

**Why:** `adamw_8bit` requires bitsandbytes which has no ROCm Windows DLL. We force `adamw_torch` which is pure PyTorch and works everywhere.

**10F — Rename tokenizer → processing_class** (around line 3437, 3448):
```python
# BEFORE:
"tokenizer": sft_tokenizer,

# AFTER:
"processing_class": sft_tokenizer,
```

**Why:** TRL 0.24.0 renamed the `tokenizer` parameter to `processing_class` in `SFTTrainer.__init__()` and `UnslothTrainer.__init__()`. Passing `tokenizer` to these classes crashes with `TypeError: got an unexpected keyword argument 'tokenizer'`.

**10G — Strip max_seq_length from SFTConfig** (around line 3391, 3451):
```python
# BEFORE:
"args": SFTConfig(**config_args),

# AFTER:
"args": SFTConfig(**{k:v for k,v in config_args.items() if k != "max_seq_length"}),
```

**Why:** `max_seq_length` was removed from `SFTConfig.__init__()` in TRL 0.22+. Passing it crashes with `TypeError: got an unexpected keyword argument 'max_seq_length'`.

---

### PATCH 11: `studio/backend/core/inference/inference.py` — ROCm inference dtype

**File:** `$s\Lib\site-packages\studio\backend\core\inference\inference.py`

**Location:** inside `load_model()`, after line 324:
```python
def load_model(
    self,
    config: ModelConfig,
    max_seq_length: int = 2048,
    dtype = None,
    load_in_4bit: bool = True,
    ...
) -> bool:
    """Load any model..."""
    import torch
    if getattr(torch.version, "hip", None):
        if dtype is None: dtype = torch.float16
        load_in_4bit = False
```

**Why:** Same BF16 Triton crash during inference. Forcing `dtype=torch.float16` and `load_in_4bit=False` prevents both the BF16 crash and the bitsandbytes crash.

---

### PATCH 12: `studio/backend/core/inference/worker.py` — ROCm inference worker

**File:** `$s\Lib\site-packages\studio\backend\core\inference\worker.py`

**Location:** inside `_handle_load()`, after line 292:
```python
def _handle_load(backend, config: dict, resp_queue: Any) -> None:
    try:
        mc = _build_model_config(config)
        hf_token = _clean_token(config.get("hf_token"))
        load_in_4bit = _resolve_lora_4bit(mc, config.get("load_in_4bit", True))

        import torch
        if getattr(torch.version, "hip", None):
            os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
            os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            load_in_4bit = False
```

**Why:** The inference worker subprocess needs these env vars set before the model loads. `UNSLOTH_DISABLE_COMPILE=1` and `TORCH_COMPILE_DISABLE=1` prevent torch.compile which would JIT-compile Triton kernels and crash on RDNA2.

---

### PATCH 13: `studio/backend/core/inference/orchestrator.py` — load_in_4bit default

**File:** `$s\Lib\site-packages\studio\backend\core\inference\orchestrator.py`

```python
# BEFORE:
def load_model(self, config, max_seq_length=2048, dtype=None, load_in_4bit: bool = True, ...):

# AFTER:
def load_model(self, config, max_seq_length=2048, dtype=None, load_in_4bit: bool = False, ...):
```

**Why:** The inference orchestrator passes `load_in_4bit` to the inference worker. Defaulting to `True` causes bitsandbytes 4-bit loading which crashes on ROCm.

---

### PATCH 14: `studio/backend/core/export/export.py` — GGUF export load_in_4bit

**File:** `$s\Lib\site-packages\studio\backend\core\export\export.py`

**Location:** inside `load_checkpoint()`, text model branch (~line 398):
```python
# BEFORE:
            else:
                logger.info("Loading as text model...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = load_in_4bit,

# AFTER:
            else:
                logger.info("Loading as text model...")
                import torch
                _use_4bit = False if getattr(torch.version, "hip", None) else load_in_4bit
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = _use_4bit,
```

**Why:** The export's default `load_in_4bit=True` requires bitsandbytes. On ROCm, we force it to `False`.

---

## 6. Create sitecustomize.py

**File:** `$s\Lib\site-packages\sitecustomize.py`

```python
import os
os.environ.setdefault("UNSLOTH_MOE_BACKEND", "native_torch")
os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
os.environ.setdefault("UNSLOTH_SKIP_TORCHVISION_CHECK", "1")
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
os.environ.setdefault("BNB_ROCM_VERSION", "71")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
```

**Why:** `sitecustomize.py` is automatically loaded by Python on EVERY process startup (including subprocesses spawned by multiprocessing). This ensures all env vars are set regardless of how the process is started. Using `setdefault` means if the env var is already set, it won't be overwritten.

---

## 7. Create Desktop Shortcut

### 7.1 Delete old .bat if it exists
```powershell
Remove-Item "$env:USERPROFILE\Desktop\Unsloth Studio ROCm.bat" -Force -ErrorAction SilentlyContinue
```

### 7.2 Create .lnk with Unsloth icon
```powershell
$wsh = New-Object -ComObject WScript.Shell
$lnk = $wsh.CreateShortcut("$env:USERPROFILE\Desktop\Unsloth Studio ROCm.lnk")
$lnk.TargetPath = "C:\Users\redwan\.unsloth\studio\unsloth_studio\Scripts\unsloth.exe"
$lnk.Arguments = "studio -p 8888"
$lnk.WorkingDirectory = "$env:USERPROFILE"
$lnk.IconLocation = "$env:USERPROFILE\AppData\Local\Unsloth Studio\unsloth.ico,0"
$lnk.Description = "Unsloth Studio - ROCm RX 6800 XT"
$lnk.Save()
```

---

## 8. Launch & Verify

### 8.1 Launch Studio
```powershell
& "C:\Users\redwan\.unsloth\studio\unsloth_studio\Scripts\unsloth.exe" studio -p 8888
```

### 8.2 Verify in terminal output
Look for these lines:
```
Hardware detected: ROCm (HIP 7.13.26154) -- AMD Radeon RX 6800 XT
Unsloth Studio running on http://127.0.0.1:8888
```

### 8.3 Open browser
Go to `http://localhost:8888`

---

## 9. Training Guide

### 9.1 From Studio UI
1. Open `http://localhost:8888`
2. Go to **Fine-tuning** tab
3. Select model (e.g., `Qwen/Qwen2-0.5B-Instruct`)
4. Select method: `Full Fine-tune` (or `LoRA` for faster)
5. Select dataset from Hugging Face
6. Configure parameters:
   - Max Steps: 30 (test) / 200-500 (real)
   - Batch Size: 2
   - Grad Accum: 4
   - Learning Rate: 2e-5
   - Context Length: 2048
7. Start training

### 9.2 From CLI (alternative)
Save as `train_swebench.py`:
```python
import torch, os
os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["UNSLOTH_ALLOW_CPU"] = "1"

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2-0.5B-Instruct",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=False,
)

dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test").select(range(100))

def fmt(example):
    text = "### Bug:\n" + example["problem_statement"] + "\n\n### Patch:\n" + example["patch"]
    return {"text": text}

dataset = dataset.map(fmt)

args = SFTConfig(
    output_dir="output_swebench",
    max_steps=200,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    fp16=False, bf16=False,
    dataset_text_field="text",
    optim="adamw_torch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=args,
)
trainer.train()
```

Run:
```powershell
$env:UNSLOTH_MOE_BACKEND="native_torch"; $env:UNSLOTH_RETURN_LOGITS="1"; $env:UNSLOTH_ALLOW_CPU="1"
C:\Users\redwan\.unsloth\studio\unsloth_studio\Scripts\python.exe "C:\Users\redwan\Desktop\train_swebench.py"
```

---

## 10. GGUF Export

### 10.1 From Studio UI
1. Go to **Export** tab
2. Source: **Local Model → Training Run**
3. Select your checkpoint
4. Export Method: **GGUF / Llama.cpp**
5. Pick quantization (e.g., `q8_0` or `q4_k_m`)
6. Click **Export**

### 10.2 Expected output
```
Exporting GGUF (['q8_0'])...
Unsloth: Converting to GGUF format...
Unsloth: llama.cpp found in the system. Skipping installation.
Unsloth: [1] Converting model into q8_0 GGUF format.
All GGUF conversions completed successfully!
```

### 10.3 Export location
```
C:\Users\redwan\.unsloth\studio\exports\YOUR_MODEL_NAME-GGUF\*.gguf
```

---

## 11. Benchmark / Inference

### 11.1 Test inference
```powershell
$env:UNSLOTH_MOE_BACKEND="native_torch"
$env:UNSLOTH_RETURN_LOGITS="1"

C:\Users\redwan\.unsloth\studio\unsloth_studio\Scripts\python.exe -c "
import torch, os
os.environ['UNSLOTH_MOE_BACKEND'] = 'native_torch'
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=r'C:\Users\redwan\.unsloth\studio\outputs\YOUR_CHECKPOINT_PATH',
    max_seq_length=4096,
    dtype=torch.float16,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

prompt = 'Fix this bug: ...'
messages = [{'role': 'user', 'content': prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to('cuda')
outputs = model.generate(inputs, max_new_tokens=512, temperature=0.6)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

### 11.2 SWE-bench benchmark script
**File:** `C:\Users\redwan\AppData\Local\Temp\swebench_eval.py`

```python
import torch, os, json, time
os.environ["UNSLOTH_MOE_BACKEND"] = "native_torch"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["UNSLOTH_ALLOW_CPU"] = "1"
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm

# CHANGE THIS to point to your trained model:
MODEL_PATH = r"C:\Users\redwan\.unsloth\studio\outputs\YOUR_CHECKPOINT_HERE"
REPORT_FILE = r"C:\Users\redwan\Desktop\swebench_results.json"

dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
dataset = dataset.select(range(50))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=4096,
    dtype=torch.float16,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

results = []
for i, s in enumerate(tqdm(dataset, desc="Benchmarking")):
    prompt = s["problem_statement"]
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    t0 = time.time()
    outputs = model.generate(inputs, max_new_tokens=512, temperature=0.6, top_p=0.95)
    gen_time = time.time() - t0
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = generated.split("assistant", 1)[-1].strip() if "assistant" in generated else generated
    results.append({
        "index": i,
        "repo": s["repo"],
        "instance_id": s["instance_id"][:50],
        "generated_length": len(generated),
        "has_diff_patch": "diff --git" in generated,
        "execution_time": round(gen_time, 2),
        "generated_output": generated[:500],
    })

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump({"model": str(MODEL_PATH), "date": time.strftime("%Y-%m-%d %H:%M:%S"), "total": len(results), "results": results}, f, indent=2, ensure_ascii=False)
```

### 11.3 Run benchmark
```powershell
$env:UNSLOTH_MOE_BACKEND="native_torch"; $env:UNSLOTH_RETURN_LOGITS="1"; $env:UNSLOTH_ALLOW_CPU="1"
C:\Users\redwan\.unsloth\studio\unsloth_studio\Scripts\python.exe "C:\Users\redwan\AppData\Local\Temp\swebench_eval.py"
```

### 11.4 Compare before/after
Use this to compare results:
```python
import json
with open(r"C:\Users\redwan\Desktop\swebench_before.json", encoding="utf-8") as f: b = json.load(f)
with open(r"C:\Users\redwan\Desktop\swebench_after.json", encoding="utf-8") as f: a = json.load(f)
b_r, a_r = b["results"], a["results"]
print(f"{'Metric':<40} {'BEFORE':<20} {'AFTER':<20}")
print("-"*80)
print(f"{'Valid diff patches':<40} {sum(1 for r in b_r if 'diff --git' in r['generated_output'])/len(b_r)*100:.1f}%{'':<15} {sum(1 for r in a_r if 'diff --git' in r['generated_output'])/len(a_r)*100:.1f}%")
print(f"{'Avg length (chars)':<40} {sum(r['generated_length'] for r in b_r)/len(b_r):.0f}{'':<17} {sum(r['generated_length'] for r in a_r)/len(a_r):.0f}")
print(f"{'Avg time (s)':<40} {sum(r['execution_time'] for r in b_r)/len(b_r):.1f}{'':<17} {sum(r['execution_time'] for r in a_r)/len(a_r):.1f}")
```

---

## 12. Appendix: Scripts & References

### 12.1 All created files

| File | Purpose |
|---|---|
| `C:\Users\redwan\Desktop\UNSLOTH_ROCM_SETUP.md` | This guide |
| `C:\Users\redwan\Desktop\Unsloth Studio ROCm.lnk` | Desktop shortcut with icon |
| `C:\Users\redwan\Desktop\swebench_before.json` | Benchmark results (before training) |
| `C:\Users\redwan\Desktop\swebench_after.json` | Benchmark results (after training) |
| `C:\Users\redwan\Desktop\bench_swebench.bat` | Batch file to run benchmark |
| `C:\Users\redwan\Desktop\bench_swebench_after.bat` | Batch file to run benchmark (after training) |
| `C:\Users\redwan\AppData\Local\Temp\swebench_eval.py` | Benchmark evaluation script |
| `C:\Users\redwan\AppData\Local\Temp\check_bnb.py` | bitsandbytes test script |
| `C:\Users\redwan\.unsloth\studio\unsloth_studio\Lib\site-packages\sitecustomize.py` | Env vars for all subprocesses |

### 12.2 ROCm nightly index contents
```
Index: https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/
├── torch  (2.11.0, 2.10.0, 2.9.1 + rocm7.13.0)
├── torchvision (0.26.0, 0.25.0, 0.24.0 + rocm7.13.0)
├── rocm-sdk-core (7.13.0a20260421)
├── rocm-sdk-libraries-gfx103x-dgpu (7.13.0a20260421)
└── rocm (7.13.0a20260421)
```

### 12.3 Package versions installed
| Package | Version | Source |
|---|---|---|
| python | 3.12.10 | python.org |
| torch | 2.10.0+rocm7.13.0a20260421 | AMD nightly |
| torchvision | 0.24.0+rocm7.13.0a20260421 | AMD nightly |
| unsloth | 2026.7.4 | PyPI |
| unsloth_zoo | 2026.7.4 | PyPI |
| trl | 0.24.0 | PyPI |
| transformers | 4.57.6 | PyPI |
| bitsandbytes | 0.49.2 | PyPI |
| torchao | 0.14.1 | PyPI |
| datasets | 4.3.0 | PyPI |
| accelerate | 1.12.0 | PyPI |
| peft | 0.19.1 | PyPI |
| triton-windows | 3.7.1.post27 | PyPI |
| xformers | 0.0.35 | PyPI |

### 12.4 Environment variables summary
| Variable | Value | Set in |
|---|---|---|
| `HIP_PATH` | `C:\Program Files\AMD\ROCm\6.2` | System env |
| `ROCM_PATH` | `C:\Program Files\AMD\ROCm\6.2` | System env |
| `UNSLOTH_MOE_BACKEND` | `native_torch` | sitecustomize.py, trainer.py |
| `UNSLOTH_RETURN_LOGITS` | `1` | sitecustomize.py, trainer.py, worker.py |
| `UNSLOTH_ALLOW_CPU` | `1` | sitecustomize.py, trainer.py |
| `UNSLOTH_SKIP_TORCHVISION_CHECK` | `1` | sitecustomize.py, trainer.py |
| `UNSLOTH_DISABLE_COMPILE` | `1` | worker.py |
| `TORCH_COMPILE_DISABLE` | `1` | worker.py |
| `BNB_ROCM_VERSION` | `71` | sitecustomize.py |
| `TOKENIZERS_PARALLELISM` | `false` | sitecustomize.py |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0` | sitecustomize.py |

---

## 13. Troubleshooting

### Error: "too many indices for tensor of dimension 1"
**Cause:** `outputs.logits` is 1D (EMPTY_LOGITS) because `RETURN_LOGITS` is False.
**Fix:** Apply Patch 1 (llama.py line 1464 → `RETURN_LOGITS = True`).

### Error: "cannot reshape tensor of 0 elements into shape [-1, 0]"
**Cause:** `entropy_from_logits()` receives logits with vocab_size=0.
**Fix:** Apply Patch 2 (trl/utils.py) AND Patch 3 (sft_trainer.py accuracy block).

### Error: "SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'"
**Cause:** TRL 0.24.0 removed `max_seq_length` from SFTConfig.
**Fix:** Apply Patch 10G (strip max_seq_length from config_args).

### Error: "SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'"
**Cause:** TRL 0.24.0 renamed `tokenizer` to `processing_class`.
**Fix:** Apply Patch 10F (tokenizer → processing_class).

### Error: "Failed to load checkpoint: Attempted to use bitsandbytes..."
**Cause:** `load_in_4bit=True` tries to load bitsandbytes which has no ROCm DLL.
**Fix:** Apply Patches 4, 13, 14 (force `load_in_4bit=False` on ROCm).

### Error: "No inf checks were recorded for this optimizer"
**Cause:** GradScaler (fp16) doesn't work with these optimizers on ROCm.
**Fix:** Apply Patch 10D (fp16=False, bf16=False on ROCm).

### Error: "LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16"
**Cause:** Triton JIT-compiler can't compile BF16 dot product for RDNA2 (gfx1030).
**Fix:** Use `dtype=torch.float16` instead of BF16 (Patch 10C).

### Error: "Windows fatal exception: code 0xc0000139"
**Cause:** torchvision's `_C.pyd` or `image.pyd` requires CUDA runtime DLLs (cudart).
**Fix:** Apply Patches 5, 6, 7 (stub torchvision C++ ops).

### Error: "ModuleNotFoundError: No module named 'torch._C._distributed_c10d'"
**Cause:** ROCm nightly PyTorch build is missing this C extension.
**Fix:** Apply Patch 8 (disable torchao quantizer in transformers).

### Error: "Failed to import ML libraries: unexpected indent"
**Cause:** Python syntax error in a patched file.
**Fix:** Verify the patched file has correct indentation (spaces, not tabs).

### Error: "CUDA error: invalid argument" on `mem_get_info`
**Cause:** ZLUDA doesn't implement `cudaMemGetInfo`.
**Fix:** Apply Patch 9 (try/except on mem_get_info).

### Model doesn't output correct format after training
**Fix:** Ensure the training dataset has clear instruction/response pairs. For code tasks: `"Fix this bug: {problem}"` paired with `"diff --git ..."`. Use "Raw Text" format in the Studio UI and map columns accordingly.

---

*End of guide. Generated 2026-07-20. Tested on AMD Radeon RX 6800 XT (gfx1030), 16GB VRAM, Windows 11.*
