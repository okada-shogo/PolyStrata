# Inkling 2 Mini

Back to [README.md](../../README.md)

**Inkling 2 Mini** by Thinking Machines: a file of 9 to 18 GB. It is experimental here. It has run on Linux with an
RTX 5090, on Windows with an 8 GB laptop card and on a Mac with an M5 Ultra (a beta there), one request at a time. It
reads pictures too. Its 24 layers of weights run as 48: the twelve in the middle run three
times for every token. Forty of the 48 look back 1,024 tokens only, so a long conversation takes little VRAM:
32,768 bytes a token.

## How fast is it?

Measured on an RTX 5090 (32 GB) with a Xeon Platinum 8470Q (25 vCPU) and 90 GB of RAM, on Linux. The card holds
every expert of all three sizes. Each run writes 256 tokens, the most likely one each time, at a 32K context, and each
number is the range of three runs.

| Size | Writes after a 19-token prompt | Writes after a 10,081-token prompt | Reads that prompt |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 111-112 tokens/s | 109 tokens/s | 3,290-3,300 tokens/s |
| **UD-Q3_K_XL** | 114 tokens/s | 110-112 tokens/s | 3,360-3,450 tokens/s |

llama.cpp on the same card, run in turns with the above on the same prompts (commit 29b52d3 with the fix of its
pull request 32704, `llama-server -ngl 99`):

| Size | Writes after a 19-token prompt | Writes after a 10,081-token prompt | Reads that prompt |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 82-83 tokens/s | 76-77 tokens/s | 1,540-1,550 tokens/s |
| **UD-Q3_K_XL** | 84-86 tokens/s | 77-78 tokens/s | 1,510-1,530 tokens/s |

Both runs print the three likeliest tokens of every step, which the comparison below reads; that costs a little.
Without it, UD-Q4_K_XL writes 127-129 tokens/s after the short prompt and 124 after the long one, and reads the long
one at 3,280-3,320 tokens/s (six runs each); the README's benchmark measures 128.

The two start with the same tokens. For UD-Q4_K_XL the first 13 after the short prompt and the first 14 after the
long one are the same, and they part at a step whose two leading candidates are nearly as likely (0.35 against 0.34,
and 0.028 against 0.028). For UD-Q3_K_XL the first 13 and the first 40 are the same.

On an RTX 30 card or newer, UD-Q2_K_XL's MXFP4 experts (the down of one layer) are multiplied on the int8 tensor
cores while it reads a prompt, with the prompt's numbers rounded to 8 bits, as the file's other experts already are.
On an RTX 3090 that reads a 2,048-token prompt 3.1% faster (2,068 against 2,005 tokens/s) and an 8,192-token one 1.0%
faster. Against a run with every product in halves, it moved the answers no more than the other int8 products do on
a 302- and an 8,192-token prompt (one prompt each). `STRATA_PREFILL_MXFP4=fp16` keeps them on halves, as before.

On an RTX 50 card the engine multiplies the experts kept in NVFP4 (most of them, in both sizes) on the FP4 tensor
cores while it reads a prompt, with the prompt's numbers rounded to about 8 bits instead of 11. That reads a prompt
1.5 to 1.9 times as fast as multiplying them as halves (`STRATA_PREFILL_NVFP4=fp16`). It moves the answers a little:
written after the halves' greedy answers to eight prompts of 19 to 10,124 tokens, 256 tokens each, it gives the same
most likely token at 98.5% of the steps (95.3% to 100% a prompt); the halves run again give 100%. A card that holds
only part of the experts was not measured. Three runs each, 32 tokens written after the prompt:

| Size | Prompt | Halves (`fp16`) | FP4 (the default on RTX 50) |
| --- | ---: | ---: | ---: |
| **UD-Q4_K_XL** | 2,664 tokens | 2,450-2,490 tokens/s | 3,800-3,840 tokens/s |
| | 10,081 tokens | 3,200-3,220 tokens/s | 5,390-5,410 tokens/s |
| | 20,162 tokens | 3,250-3,260 tokens/s | 5,500-5,510 tokens/s |
| **UD-Q3_K_XL** | 2,664 tokens | 2,570-2,600 tokens/s | 3,750-3,880 tokens/s |
| | 10,081 tokens | 3,340-3,360 tokens/s | 5,360-5,400 tokens/s |
| | 20,162 tokens | 3,390-3,400 tokens/s | 5,480-5,510 tokens/s |

On an RTX 20 card (Turing) the engine multiplies the experts as halves while it reads a prompt. With `--prompt-int8`
it multiplies them on the int8 tensor cores, as it does on RTX 30 and newer, with the prompt's numbers rounded to 8
bits with a scale for every 32 (`STRATA_PF_I8MMA=0` turns the int8 products off on every card). It is not the
default because it moves the answers more: measured on a Quadro RTX 4000 (8 GB),
which holds a third to two thirds of the experts, written after the halves' greedy answers to three prompts of 302
to 8,192 tokens, 64 tokens each, with three files (the two below and an IQ3_S), it gives the same most likely token
at 560 of the 576 steps (97.2%). Prompt speed on that card, three runs each:

| Size | Prompt | Halves (the default on RTX 20) | int8 (`--prompt-int8`) |
| --- | ---: | ---: | ---: |
| **IQ4_NL** | 2,048 tokens | 329-338 tokens/s | 419-447 tokens/s |
| | 8,192 tokens | 355-361 tokens/s | 456-461 tokens/s |
| **UD-Q2_K_XL** | 2,048 tokens | 361-363 tokens/s | 434-439 tokens/s |
| | 8,192 tokens | 347-348 tokens/s | 411-412 tokens/s |

On an RTX 30 card or newer, most of a prompt's products run on the int8 tensor cores, with the prompt's numbers
rounded to 8 bits. UD-Q2_K_XL's IQ1_S experts (the gate and up of 6 layers) are the exception: they stay on halves
unless `STRATA_PREFILL_IQ1S=int8` is set. On an RTX 3090 that setting reads a 2,048-token prompt 7.7% faster (2,159
against 2,005 tokens/s) and an 8,192-token one 6.7% faster, but on a 302-token prompt it moved the answers more for
each matrix than the other experts' int8 products do, so it is opt-in until more prompts are measured.

On a laptop with an RTX 4070 Laptop (8 GB), a Core i7-14700HX and 32 GB of RAM, on Windows, UD-Q3_K_XL has 1,216 of
its 3,072 experts on the card and the rest are computed on the CPU. On a card that does not hold every expert, setup
gives Inkling 2 Mini three settings that read fewer bytes for each token: `--attn-requant q5_0` (the attention's
matrices in 5 bits instead of 8), `--loop-bf16` (the loop controllers in BF16 instead of F32) and `--tail-skip 4`.
A card that holds the whole model gets none of them. Expert swaps stay on: with `--expert-swaps 0` the experts the
card holds stay the ones it loaded, fewer of a token's experts are found there, and the CPU computes more of them
(16-23% slower writing after a 302-token prompt on an RTX 4060 Ti 16 GB, a Quadro RTX 4000 8 GB and an RTX 2070
Super 8 GB, about the same after 8,192). On the laptop, the fastest of three runs, 256 tokens written each; the last
column is how often the most likely token is the same as without the settings, written after their greedy answer to
the 4,526-token prompt:

| UD-Q3_K_XL | Writes after a 25-token prompt | Writes after a 4,526-token prompt | Same most likely token |
| --- | ---: | ---: | ---: |
| None of the settings | 37.8 tokens/s | 41.8 tokens/s | 99.6% (run again) |
| Setup's three settings | 47.7 tokens/s | 50.0 tokens/s | 98.0% |
| The three and `--expert-swaps 0` | 53.6 tokens/s | 52.7 tokens/s | 95.7% |

On the laptop `--expert-swaps 0` writes faster, but it moves the answers more, so setup leaves it out.

A prompt is read a group of calls at a time through each layer run, so the experts that are not on the card cross
the bus once for the group instead of once for every call and every pass of the looped layers.

`--kv fp8` keeps the attention caches in 8 bits instead of 16, which leaves about 0.7 GB more VRAM for the experts at
a 32K context.

On a Mac it runs on strata-poly's Metal engine, as a beta. Measured on an Apple M5 Ultra with 256 GB of unified memory,
UD-Q3_K_XL, the README's benchmark on setup's settings with a 40,960-token context: 256 tokens written, the most likely
one each time. Other programs were using the GPU at the same time. Two runs of the same prompts given to the engine
alone wrote 73-74 and 60-61 tokens/s and read 2,411 tokens/s.

| UD-Q3_K_XL | Writes in a short chat | Reads a 32,787-token prompt | Writes after it |
| --- | ---: | ---: | ---: |
| M5 Ultra | 74 tokens/s | 2,440 tokens/s | 61 tokens/s |

Given the 128 tokens the CUDA engine wrote after a 4,526-token prompt (UD-Q3_K_XL), it picks the same most likely token
at 126 of the 128 steps. UD-Q2_K_XL has run there too (73 tokens/s in the short chat). UD-Q4_K_XL has not been measured
on a Mac. Pictures are read on the Mac's CPU, because the picture tower has no Metal kernels: a screenshot read as 984
rows took 3.4 s.

## What you need

| | |
| --- | --- |
| **Graphics card** | One **NVIDIA** card with **8 GB of VRAM or more**. It has run on an RTX 5090 (32 GB) and on an RTX 4070 Laptop (8 GB). About 3.7 GB are taken before any expert at a 32K context. Or an **Apple Silicon Mac** (beta): it has run on an M5 Ultra. |
| **RAM** | About **21, 17 or 13 GB** beside an 8 GB card for UD-Q4_K_XL, UD-Q3_K_XL and UD-Q2_K_XL, the installer's estimate. A 32 GB card holds any of them whole. On a Mac, setup takes UD-Q3_K_XL from 32 GB of unified memory and UD-Q2_K_XL below that. |
| **Disk** | 9.1 to 17.6 GB for the model file. |
| **System** | Linux or Windows 10 / 11, and a current NVIDIA driver; or macOS 15 or newer. |

Not run yet: an AMD card, two cards (`--expert-gpu`, built but not run; `--layer-split`, below, checked with two
parts on one card only), UD-Q4_K_XL on the 8 GB card and UD-Q4_K_XL
on a Mac. Pictures go through Unsloth's mmproj-F16.gguf, read on the CPU (`--vision cpu`) or the card; on a Mac, on
the CPU. The draft file is an
option (`--draft-model`). The engine takes one request at a time
for this model, and the installer writes `"parallel": 1`. Its license is Apache 2.0.

## Install

```sh
./setup.sh --family inkling
```

Add `--model UD-Q3_K_XL` or `--model UD-Q2_K_XL` for a smaller size.

## Which model should I pick?

| Size | What it is | Download | RAM beside an 8 GB card |
| --- | --- | ---: | ---: |
| **UD-Q4_K_XL** | 4-bit experts, 8-bit where they write ([Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)) | 17.6 GB | about 21 GB |
| **UD-Q3_K_XL** | 3- and 4-bit experts mixed | 13.6 GB | about 17 GB |
| **UD-Q2_K_XL** | 1- to 2-bit experts: the smallest | 9.1 GB | about 13 GB |

The installer downloads the files under the repository's `Mainline_Support` folder, the layout llama.cpp runs
(`inkling2_loop`). The files at the top of the repository are another layout, which the engine refuses by name.

## Two or more cards

`--layer-split` puts the first layers on one card and the rest on the next, as for the other strata-poly models.
Each card keeps its layers' weights, attention caches and expert cache. The loop's twelve layers (6 to 17, which
run three times a token) stay on one card, so the second card starts at layer 1 to 6 or 18 to 22; a number inside
the loop is refused. `"auto"` shares the layers by each card's free VRAM and takes 6 or 18, whichever is nearer.
Once a token, a card hands the next one the token's residual streams and the loop's inputs, 64.5 KB. The DSpark
drafter (`--draft-model`) runs on one card only.

## How does it work?

A reply is a list of typed blocks: thinking, text, or a tool call with its arguments as JSON. The server turns
them into the same reasoning, content and tool calls as for the other models, so the OpenAI and Anthropic APIs work
as they do elsewhere. A request that asks for no thinking has its prompt end inside the text block, because the
model otherwise opens a thinking block now and then. More in
[BACKENDS.md](../BACKENDS.md#inkling-2-mini-one-card).
