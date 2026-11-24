# Inkling 2 on AMD: preview branch `inkling2-amd-preview-2`

This branch is PolyStrata main (36e654b1b) with our open AMD and VRAM pull requests merged into it. These PRs are
not on main yet, so this is a preview: use main for anything you rely on. It replaces the old
`inkling2-amd-preview` (3bc4042c, built on the closed #200), which stays as it was.

Tested on one card only: RX 9070 XT 16 GB (gfx1201), native Linux (Fedora, in-tree amdgpu, ROCm 7.2.4). Not tested
by us: RX 9060 XT (gfx1200), Windows, ROCm 10.x.

## What is in it

| PR | What it does |
|---|---|
| #285 | `--bench-report`: a built-in profiler and a report on what limits a card |
| #288 | fewer decode kernels when some experts are computed on the CPU |
| #290 | all experts in pinned RAM when the RAM has room (faster refills) |
| #291 | prompt kernels loaded at startup, not at the first prompt |
| #292 + #289 step A (`port/hip-int8-auto`) | prompt products on RDNA3/RDNA4 int8 matrix cores, on by default (see below) |
| #293 | AMD: a simpler decode graph (faster decode under WSL2; no change on native Linux) |
| #294 | AMD: fewer launches for the attention projections |
| #295 | AMD: the clip and KV-store kernels fused |
| #299 | faster expert-cache refill after long prompts |
| #271 | opt-in prefetch of the next layer's experts (`STRATA_INK_PRESTAGE=1`; CUDA only, it does nothing on AMD) |
| #301 | a graph that does not fit in VRAM no longer fails the request |
| #302 | AMD under WSL2: no 4 s stall when the expert cache shrinks |
| #306 | AMD: the GPU asks the CPU for the experts it lacks without a host round trip a layer (`STRATA_POLY_BELL=0`: off) |
| #316, #317, #321 | DSpark drafter: less VRAM, and Q4_0 / IQ4_NL drafter files |
| #319 | `--vram-elastic`: the expert cache gives VRAM back to other programs while serving, and takes it back |
| #320 | `--prompt-batch auto` (the default): the prompt batch sized from free VRAM |
| #322 | AMD on Linux: the expert staging buffer trimmed to what the cache misses (more experts fit in VRAM) |
| `port/hip-win-no-lend` | AMD on Windows: no prompt-buffer lending (the crash in amdhip64_7.dll of issue #305) |

## Build

Linux, with a system ROCm in `/opt/rocm` (gfx1201 = RX 9070 / 9070 XT / R9700, gfx1200 = RX 9060 / 9060 XT):

```
git clone -b inkling2-amd-preview-2 https://github.com/okada-shogo/PolyStrata.git
cd PolyStrata
cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_NATIVE_EXPERTS=ON
cmake --build build --target strata-poly -j8
```

Use `-DCMAKE_HIP_ARCHITECTURES=gfx1200` on an RX 9060 XT, or `gfx1200;gfx1201` for both. Setup
(`./setup.sh --family inkling`) also builds this checkout; it uses a system ROCm 7 or newer, else AMD's TheRock
wheels (7.10).

Windows: `START-HERE.bat --setup --family inkling` compiles `strata-poly.exe` from this checkout with AMD's TheRock
Windows wheels (ROCm 10.2). **We have not built or run this branch on Windows.** main's Windows HIP build works
(#323), and this branch adds no Windows-only code. Please report if it does not build.

## Run it for best speed

int8 prompt products are on by default (auto: the int8 kernel for every type it takes, RDNA3 and RDNA4). You don't
need to set anything for it. It rounds the activations to int8 with a scale per 32 values: the same rounding class as
llama.cpp's MMQ, PolyStrata 0.1.45 and CUDA's default on Ampere and newer. `STRATA_POLY_HIP_INT8=0` gives main's path.

**Setup on any OS** adds `--attn-requant q5_0 --loop-bf16 --tail-skip 4` whenever the card cannot hold every expert of
the size (for example IQ4_NL or UD-Q4_K_XL on a 16 GB card). They are for 8 GB cards, and `--tail-skip` skips experts,
so it changes the answers. Run setup with `--inkling-small-card off` (setup keeps that choice), or remove the three
flags from the run config's `"args"`.

**Linux, native (fastest).** ROCm 7.2.x recommended. No small-card flags. The default window and `--prompt-batch auto`
need no flags either:

```
./build/strata-poly --model Inkling-2-Mini-IQ4_NL.gguf --serve --max-context 32768 --threads $(( $(nproc --all) / 2 - 1 ))
```

`--threads` is the physical cores minus one. On a 16 GB card add the DSpark drafter
([dspark-Inkling-2-Mini-Q8_0.gguf](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/dspark-Inkling-2-Mini-Q8_0.gguf), 400 MB):

```
  --draft-model dspark-Inkling-2-Mini-Q8_0.gguf
```

On our RX 9070 XT (with an earlier combined build, without #306) it wrote 121-124 tok/s on IQ4_NL and 143-155 on
UD-Q2_K_XL after 2K-8K prompts, against 64-73 without. After a short prompt it can be a few percent slower (IQ4_NL
after 302: 68.0 against 70.3). With a desktop on the same card, add `--vram-elastic`: the expert cache then gives VRAM back when
the desktop takes more.

**Windows.** `START-HERE.bat --setup --family inkling --inkling-small-card off` builds and writes the run config.
Setup installs AMD's TheRock ROCm 10.2. We measured main's prompts much slower with it (below). The int8 default
may avoid most of that, but this is not confirmed. The ROCm 10 work is not in this preview.

**Reporting.** So that reports compare, please post these four runs (one request each, the same prompt) as one
table:

| Run | Settings | Prompt (tok/s) | Decode (tok/s) |
|---|---|---:|---:|
| 1 | default | | |
| 2 | `STRATA_POLY_HIP_INT8=0` | | |
| 3 | default + `--draft-model` | | |
| 4 | default, `STRATA_POLY_BELL=0` | | |

Add the card, OS, ROCm version, model file and the "N of 3072 routed experts in VRAM" line of the log.

## Speed to expect

RX 9070 XT 16 GB, native Linux (Fedora, in-tree amdgpu), ROCm 7.2.4, Ryzen 7 7800X3D, `--serve --threads 11
--vram-reserve-mib 700 --kv fp16 --max-context 32768`, the same flags on both sides, mean of 3 runs with the order
rotated, this exact tree. main is 36e654b1b with its defaults. Experts in VRAM: IQ4_NL 2,350 (main) and 2,494 (this branch) of 3,072;
UD-Q2_K_XL 3,058 on both.

Prompt tok/s:

| Card | Model | Measure | main | this branch | Change |
|---|---|---|---:|---:|---:|
| RX 9070 XT 16 GB | IQ4_NL | prompt 2,048 (tok/s) | 1,121.7 | 1,685.8 | +50.3% |
| RX 9070 XT 16 GB | UD-Q2_K_XL | prompt 2,048 (tok/s) | 1,405.6 | 2,583.8 | +83.8% |

Decode tok/s (256 tokens written):

| Card | Model | Measure | main | this branch | Change |
|---|---|---|---:|---:|---:|
| RX 9070 XT 16 GB | IQ4_NL | decode after 302 (tok/s) | 63.9 | 78.0 | +22.1% |
| RX 9070 XT 16 GB | IQ4_NL | decode after 8,192 (tok/s) | 60.8 | 73.2 | +20.4% |
| RX 9070 XT 16 GB | UD-Q2_K_XL | decode after 302 (tok/s) | 65.7 | 84.0 | +27.9% |
| RX 9070 XT 16 GB | UD-Q2_K_XL | decode after 8,192 (tok/s) | 62.2 | 77.9 | +25.2% |

The RX 9060 XT (gfx1200) has half the compute units: expect lower numbers there.

## ROCm version and rocBLAS

ROCm 10 itself is not slower: on native Linux, main reads a 2,048-token prompt at 1,372 tok/s on ROCm 10.2 and 1,286
on 7.2.4 (RX 9070 XT, IQ4_NL). What matters is whether rocBLAS hands its matrix products to hipBLASLt. With
`ROCBLAS_USE_HIPBLASLT=0` main's prompt drops 57% on both versions, and some distro packages build rocBLAS without
hipBLASLt (Arch's, which CachyOS uses, has `-DBUILD_WITH_HIPBLASLT=OFF`). This branch runs the prompt expert products
on its own int8 kernels, not rocBLAS, so it depends much less on that; the dense products still go through rocBLAS.

On Windows, one build of main read 2,048-token prompts at 307 tok/s against 1,426 on native Linux on the same card,
but 8,192-token prompts at 996: that points at per-call overhead on Windows, which the int8 path mostly avoids. We
have not measured this branch on Windows.

## Known limits

- Answers are not bit-identical to main: the int8 prompt products and the GPU-side routing of #306 round
  differently (the same classes as CUDA main). `STRATA_POLY_HIP_INT8=0` and `STRATA_POLY_BELL=0` give main's paths back.
- No images on AMD.
- Untested: Windows, gfx1200, RDNA3, ROCm 10.x, contexts past 32K.
- A preview: it is rebased when the PRs change, and it will be deleted once they are on main.

Please report speed with the same table (card, model, measure, tok/s) in your issue.
