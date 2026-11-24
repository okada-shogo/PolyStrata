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
| #293 | AMD: a simpler decode graph (faster decode) |
| #294 | AMD: fewer launches for the attention projections |
| #295 | AMD: the clip and KV-store kernels fused |
| #299 | faster expert-cache refill after long prompts |
| #271 | opt-in prefetch of the next layer's experts (`STRATA_INK_PRESTAGE=1`) |
| #301 | a graph that does not fit in VRAM no longer fails the request |
| #302 | AMD under WSL2: no 4 s stall when the expert cache shrinks |
| #306 | AMD: the GPU asks the CPU for the experts it lacks without a host round trip a layer (`STRATA_POLY_BELL=0`: off) |
| #316, #317, #321 | DSpark drafter: less VRAM, and Q4_0 / IQ4_NL drafter files |
| #319 | `--vram-elastic`: the expert cache gives VRAM back to other programs while serving, and takes it back |
| #320 | `--prompt-batch auto` (the default): the prompt batch sized from free VRAM |
| #322 | AMD on Linux: the expert staging buffer trimmed to what the cache misses (more experts fit in VRAM) |

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

## Recommended settings (16 GB cards)

```
./build/strata-poly --model Inkling-2-Mini-IQ4_NL.gguf --serve --max-context 32768 --threads <cores - 1>
```

- int8 prompt products are on by default here: on RDNA3 (gfx11) and RDNA4 (gfx12), every prompt product of a type
  the int8 kernel takes runs on the int8 matrix cores. main uses FP16 tiles on RDNA3, and on RDNA4 it dequantizes
  each expert to FP16 and multiplies it through rocBLAS / hipBLASLt, one call per expert, which is slow, more so on
  ROCm 10.x.
  The activations are rounded to int8 (a scale per 32 values): the same rounding class as llama.cpp's MMQ, PolyStrata
  0.1.45, and CUDA's default on Ampere and newer. `STRATA_POLY_HIP_INT8=0` turns it off (main's path).
- No small-card flags. Setup adds `--attn-requant q5_0 --loop-bf16 --tail-skip 4` on any card that does not hold every
  expert. They exist for 8 GB cards. `--tail-skip 4` skips some experts entirely, so the answers change. On a 16 GB
  card the decode below is reached without them. Use `--inkling-small-card off` with setup, or remove the three flags
  from the run config (docs/models/inkling.md).
- Linux desktop on the same card: leave the default reserve, or `--vram-elastic` so the cache gives VRAM back when
  the desktop takes more.
- Windows: setup's run config, with `--inkling-small-card off` on a 16 GB card.

## Speed to expect

RX 9070 XT 16 GB, native Linux (Fedora, in-tree amdgpu), ROCm 7.2.4, Ryzen 7 7800X3D, `--serve --threads 11
--vram-reserve-mib 700 --kv fp16 --max-context 32768`, the same flags on both sides, mean of 3 runs with the order
rotated. main is 36e654b1b with its defaults. Experts in VRAM: IQ4_NL 2,350 (main) and 2,494 (this branch) of 3,072;
UD-Q2_K_XL 3,058 on both.

Prompt tok/s:

| Card | Model | Measure | main | this branch | Change |
|---|---|---|---:|---:|---:|
| RX 9070 XT 16 GB | IQ4_NL | prompt 2,048 (tok/s) | 1,117.5 | 1,694.6 | +51.6% |
| RX 9070 XT 16 GB | UD-Q2_K_XL | prompt 2,048 (tok/s) | 1,404.4 | 2,584.1 | +84.0% |

Decode tok/s (256 tokens written):

| Card | Model | Measure | main | this branch | Change |
|---|---|---|---:|---:|---:|
| RX 9070 XT 16 GB | IQ4_NL | decode after 302 (tok/s) | 64.1 | 78.1 | +21.8% |
| RX 9070 XT 16 GB | IQ4_NL | decode after 8,192 (tok/s) | 60.8 | 73.2 | +20.4% |
| RX 9070 XT 16 GB | UD-Q2_K_XL | decode after 302 (tok/s) | 65.6 | 84.1 | +28.2% |
| RX 9070 XT 16 GB | UD-Q2_K_XL | decode after 8,192 (tok/s) | 62.1 | 77.9 | +25.4% |

The RX 9060 XT (gfx1200) has half the compute units: expect lower numbers there.

## ROCm version

We measured with ROCm 7.2.4. On the same RX 9070 XT, one build (main + #285's profiler) read prompts of 302 and
2,048 tokens 4.1x and 4.6x slower on Windows with setup's TheRock ROCm 10.2 than on native Linux with ROCm 7.2.4
(165 and 307 against 671 and 1,426 tok/s). That build ran the prompt expert products through the rocBLAS loop. This branch runs them on its own int8 kernels,
so most of that gap should go, but we have not measured ROCm 10.1 or 10.2 with them. If prompts
stay far below the table above, try ROCm 7.2.x.

## Known limits

- Answers are not bit-identical to main: the int8 prompt products and the GPU-side routing of #306 round
  differently (the same classes as CUDA main). `STRATA_POLY_HIP_INT8=0` and `STRATA_POLY_BELL=0` give main's paths back.
- No images on AMD.
- Untested: Windows, gfx1200, RDNA3, ROCm 10.x, contexts past 32K.
- A preview: it is rebased when the PRs change, and it will be deleted once they are on main.

Please report speed with the same table (card, model, measure, tok/s) in your issue.
