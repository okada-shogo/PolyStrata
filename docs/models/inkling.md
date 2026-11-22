# Inkling 2 Mini

Back to [README.md](../../README.md)

**Inkling 2 Mini** by Thinking Machines, released on 2026-11-20: a mixture of 128 experts, 8 used a token, in 24
layers. Layers 6 to 17 run three times for each token, so a token goes through 48 layers with the weights of 24. Five
layers in six look back only 1,024 tokens. PolyStrata runs [ggml-org's GGUF files](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)
and [Unsloth's UD-Q2_K_XL](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) on the strata-poly engine.

**It is experimental and new.** Support was added on 2026-11-21. Its speeds below are from one PC with an RTX 2070 Super
8 GB unless a row names a rented card. Its logits were
checked against llama.cpp's on the same files (below): the same most likely next token after each of the prompts, and
the same at 29 to 32 of 32 steps with llama.cpp's own answer fed in.

## How fast is it?

> **Measured on:** RTX 2070 Super 8 GB, Ryzen 5 3600XT, 62 GB DDR4, Ubuntu 22.04, 2026-11-22. `strata-poly --serve`
> with the server's flags (128K context, q8_0 caches), one engine a run, each from a fresh copy of the model's expert
> profile, the prompt read in a fresh conversation (text from the parity prompts repeated). Reading at 2,048, 8,192 and
> 32,768 tokens: prompt only, two runs, with the next layer's experts copied while a layer runs (below). The other
> cells: the same flags, a build before that copy (it changes reading only); 302 and 1,500 tokens two runs each with
> 256 tokens written, the longer prompts one run with 64. 1,042 of IQ4_NL's and 2,022 of UD-Q2_K_XL's 3,072 experts
> were cached in the copy's runs.

| prompt | IQ4_NL reading | IQ4_NL writing | UD-Q2_K_XL reading | UD-Q2_K_XL writing |
| --- | ---: | ---: | ---: | ---: |
| 302 tokens | 282-290 tokens/s | 38.8-40.0 tokens/s | 557-561 tokens/s | 51.1-52.3 tokens/s |
| 1,500 tokens | 521-523 tokens/s | 38.5-38.7 tokens/s | 881-888 tokens/s | 49.0-49.5 tokens/s |
| 2,048 tokens | 787-789 tokens/s | 34.0 tokens/s | 929-931 tokens/s | |
| 8,192 tokens | 947-948 tokens/s | | 888-896 tokens/s | |
| 32,768 tokens | 911-914 tokens/s | 28.7 tokens/s | 852-854 tokens/s | |
| 131,008 tokens | 659 tokens/s (199 s) | 20.6 tokens/s | | |

A prompt batch's attention runs on the tensor cores (cards of Turing or newer); writing does not (a token at a time
takes the split path). Single runs differ by up to 4.5% both ways, the means of two runs by at most 1.5%. A prompt's
read copies the next layer's experts that the card does not hold while a layer runs, and hands the card each call's
row tables by a kernel, so that they wait for no copy.

The results are the same as without those changes. With `STRATA_FREEZE_EXPERTS=1`, the 20 most likely tokens and
their probabilities at 4 steps after prompts of 2,048 and 8,192 tokens (q8_0 caches) and of 2,048 tokens (fp16)
matched for both models, in runs where both builds cached the same number of experts (that number depends on the VRAM
free at load, and runs that differ in it differ in their results with either build). Through `--serve`, also frozen,
at a 40K context with `--kv-whole` (so that a frozen 32K read has its rows), it gave the same tokens after an
8,192-token prompt with a checkpoint inside its read, after that prompt resumed with 1,985 tokens more, and after a
32,765-token prompt read in four reads that followed a read stopped at 3 s.

`STRATA_POLY_EXPERT_TILES=1` (opt-in) puts a prompt batch's expert products in one launch of tiles of their own
widths. It read prompts 6-17% faster on the RTX 2070 Super 8 GB, a rented RTX 3090 24 GB and a rented A100 40 GB. With
every expert on the card the results were the same to the bit as without it; with some experts in RAM the sums run in
another order, and the chosen token changed only where the top two were nearly tied.

On an A100 (compute capability 8.0 only) a full-attention layer whose prompt call reaches past 2,048 keys runs on a
FlashAttention-2 kernel that gives the same bits as the tensor-core kernel above. On a rented A100 40 GB (every expert in
VRAM) IQ4_NL read prompts of 8K to 128K tokens 4-6% faster with it.

On a card that holds every expert, a token written on its own runs as one CUDA graph: its 48 layer runs and their
experts are recorded once (again when the context outgrows the size they were recorded for) and replayed, so the CPU
hands the card 4 launches a token instead of 1,059 (IQ4_NL) or 1,167 (UD-Q2_K_XL). A window of guesses (`--window`
above 1) is recorded the same way, once for each number of tokens it puts through the model, and its guesses are
checked between windows as before. Not recorded: several requests at once (`--parallel`), a picture's tokens and
`--dump-layers`. `--no-graphs` turns it off; the output is the same to the bit either way. A card that holds only
some of the experts, as the 8 GB one above, runs a token launch by launch.

> **Measured on:** a rented RTX 3090 24 GB, every expert in VRAM, strata-poly alone, a 16K context, q8_0 caches, 256
> tokens written, two runs each, a build that ran a token launch by launch and this one in turn.

| writing | launch by launch | one CUDA graph |
| --- | ---: | ---: |
| IQ4_NL, after 302 tokens | 62.5-62.8 tokens/s | 69.8-70.2 tokens/s (+11%) |
| IQ4_NL, after 8,192 tokens | 60.8-61.3 tokens/s | 67.4-67.8 tokens/s |
| UD-Q2_K_XL, after 302 tokens | 71.8-72.6 tokens/s | 82.1-82.6 tokens/s (+14%) |
| UD-Q2_K_XL, after 8,192 tokens | 69.6-70.2 tokens/s | 78.7-79.4 tokens/s |

A token's launches are fewer on every card: a visit's norms and Q8_1 copies, the cache store and its clip, the
attention's gate and the mixes around it go in fewer kernels that write the same bits as the separate ones (the
kernel test compares them). With UD-Q2_K_XL on the RTX 3090 a token took 890 kernels on the card instead of 1,391
(nsys, every expert in VRAM). On CUDA a token's mixes are spread over many blocks in a cooperative launch, which the
runtime starts with all of its blocks on the card together, or else they run one block a mix;
`STRATA_INKLING_MIX_SPREAD=0` keeps them one block a mix, with the same bits.

IQ4_NL's speed depends on its expert profile (`--expert-profile`, which the server keeps up to date): it chooses the
experts the card holds. On 100 questions (50 GSM8K, 50 MMLU-Pro, greedy, medium reasoning) IQ4_NL got 49 and 44
right, UD-Q2_K_XL 48 and 41; the two gave the same answer to 92 of the 100.

About 4.6 GiB of experts fit on the 8 GB card. At most 7.5 GiB of VRAM was in use (nvidia-smi, IQ4_NL, a
31,600-token prompt). The engine's RAM was 16.5 GiB with IQ4_NL.

## What you need

| | |
| --- | --- |
| **Graphics card** | An **NVIDIA** card, or several that split its layers ([Two or more cards](#two-or-more-cards)). About 3.1 GiB of VRAM before any expert at an 8K context (measured on an 8 GB card), about 4 GiB at 32K. |
| **RAM** | About **22 GB** (IQ4_NL), **14 GB** (UD-Q2_K_XL) or **32 GB** (Q8_0) beside an 8 GB card, setup's estimate (the engine itself used 16.5 GiB with IQ4_NL, 9.1 GiB with UD-Q2_K_XL). The experts the card does not hold stay in RAM. Beside a 24 GB card the IQ4_NL file's experts fit in VRAM by the same estimate. |
| **Disk** | 17.6, 9.1 or 27.8 GB for the model file. |
| **System** | Linux and a current NVIDIA driver. On Windows it ran for one user on an earlier build (MSVC, an RTX 4060 Laptop), but it is not re-checked there, so setup asks before it installs it. |

Setup does not offer it on AMD or Intel cards or on a Mac. The engine builds and runs it on AMD cards with HIP (an
RX 9070 XT with ROCm 7.2.4, and an RX 9060 XT), but its prompt speed there is not tuned yet. Two or
more NVIDIA cards can split its layers ([Two or more cards](#two-or-more-cards)). It can answer several requests at once
(`--parallel`, see [Several requests at once](#several-requests-at-once)). Its attention caches are fp16, or q8_0 with
`--kv q8_0`, which halves them: a 128K context then takes about 2.3 GiB of VRAM when it is full, in fp16 4.3 GiB. On an
8 GB card, use q8_0 for contexts past 32K. The file allows 1,048,576 tokens and the engine sets no limit of its own, but
the cache stays in VRAM (about 17 KiB a token in q8_0; the engine's KV streaming is not wired up for this model).
Prompts were tried up to 131,008 tokens on the RTX 2070 Super 8 GB (a 128K context, q8_0 caches) and up to 1,048,576
on a rented A100 40 GB (UD-Q2_K_XL, every expert in VRAM, q8_0 caches, an earlier build: read in 1,782 s, then 17.8
tokens/s writing, 27.7 GB of VRAM at most). Pictures are optional: see [Images](#images). It writes one token at a
time unless it guesses from the conversation (`--window`, see
[Guesses from the conversation](#guesses-from-the-conversation)) or with its drafter (see
[Guesses from its DSpark drafter](#guesses-from-its-dspark-drafter)).

## Install

```bash
./setup.sh --family inkling
```

```bash
./setup.sh --family inkling --model UD-Q2_K_XL
```

```bash
./setup.sh --family inkling --model Q8_0
```

## Which model should I pick?

| Size | What it is | Download | RAM beside an 8 GB card (estimate) |
| --- | --- | ---: | ---: |
| **IQ4_NL** | 4-bit experts in layers 0 to 19 (with an importance matrix), 8-bit everywhere else | 17.6 GB | about 22 GB |
| **UD-Q2_K_XL** | Unsloth's: 2-bit experts (Q2_K, IQ1_S and Q2_0), higher where it matters. The fastest, a little less accurate | 9.1 GB | about 14 GB |
| **Q8_0** | 8-bit throughout: the closest to the model as trained | 27.8 GB | about 32 GB |

Setup takes IQ4_NL unless you name another size. UD-Q2_K_XL comes from Unsloth's repository in the architecture's
native form (`inkling2`, the loop not unrolled); the engine reads that form as it is. Q8_0 was not run. ggml-org's model card gives the KL divergence against the BF16 file on
wikitext-2: 0.0030 for Q8_0 and 0.0095 for IQ4_NL. PolyStrata did not measure that.

## Several requests at once

With `"parallel": N` in the run config (or `--parallel N` when you run setup) the engine answers up to N requests
together, as for the other models of strata-poly ([BATCHING.md](../BATCHING.md)). Each request then has its own
attention caches and convolution state. A token of every request goes through the model in one pass, so the dense
weights, and each expert that two requests both use, are read once for all of them. With the expert cache held still
(`--check-batch`, [BATCHING.md](../BATCHING.md)), every sequence got the tokens it gets alone: IQ4_NL and UD-Q2_K_XL,
fp16 and q8_0 caches, 2, 3 and 4 sequences after a 302-token prompt and 4 after a 3,000-token one, and 4 sequences
that start, join and stop at different times (`--check-stagger`: prompts of 16, 46, 375 and 3,000 tokens, read while
the others write, answers of 64 to 16 tokens, so that a step has 1 to 4 of them).

> **Measured on:** RTX 2070 Super 8 GB, Ryzen 5 3600XT, 62 GB DDR4, Ubuntu 22.04: the server's config (128K context,
> q8_0 caches, the expert profile), chats of 300 tokens started together (greedy, thinking off), two runs each; the
> server was started again for each row, on an earlier build than the speeds at the top of this page.

| Chats at once | `"parallel"` | IQ4_NL, all together | IQ4_NL, each chat | UD-Q2_K_XL, all together | UD-Q2_K_XL, each chat | Experts in VRAM (IQ4_NL / UD-Q2_K_XL) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 37.1-37.7 tokens/s | 39.6-40.2 | 46.2-47.8 | 49.6-51.2 | 1,034 / 2,005 |
| 1 | 4 | 34.5-35.0 | 36.9-37.6 | 44.3-44.9 | 46.7-47.4 | 782 / 1,547 |
| 2 | 2 | 44.6-45.5 | 23.5-24.9 | 57.7-59.7 | 30.1-32.1 | 911 / 1,777 |
| 3 | 3 | 48.9-49.0 | 17.0-18.2 | 63.2-64.5 | 21.9-23.2 | 849 / 1,663 |
| 4 | 4 | 51.2-51.9 | 13.3-14.4 | 64.1-66.5 | 16.6-18.0 | 786 / 1,547 |

"All together" is the tokens of all the chats over the time from the first request to the last answer, prompts
included. "Each chat" is a chat's writing speed as the server reports it (`timings`). On this card most experts run
on the CPU, and requests that route to different experts each need their own, so four chats together wrote 1.34 to
1.44 times as much as one alone. One chat with `"parallel": 4` was 3 to 9% slower than with 1: the slots' VRAM leaves
fewer experts on the card. A card that holds more of the experts may gain more; that was not measured.

Each slot takes VRAM from the expert cache at load: its 40 window layers' rows and its convolution state. With q8_0
caches `"parallel": 2` took about 0.5 GiB (it adds two sequences beside the first) and each slot after it about
0.27 GiB (the experts the cache held less, at the average size of a cached expert). Its full-attention rows (17 KiB a
token in q8_0) come out of the cache as its conversation grows, as the first request's do.

## Guesses from the conversation

With `--window 4` in the run config's `"args"` (`"--window", "4"`) the engine guesses what comes next where the
answer repeats the conversation: after an earlier repeat of the last three tokens or more (`--lookup N`, default 3) it
takes what followed as up to three guesses, and puts the known token and the guesses through the model together. A
guess counts when the model would have written that token itself, so the answer is the one it writes without
guesses. It is off by default: without `--window` the engine writes one token at a time, identical to `--window 1`.
The model's file has no draft block; its drafter (below) is a file of its own.

It pays where the answer copies text from the prompt, such as code given back with a change. Elsewhere it guesses
rarely and costs about nothing. On a card that holds few of the experts it gains less: most experts of the guessed
tokens run on the CPU, which takes about as long for a guessed token as for one alone.

> **Measured on:** RTX 2070 Super 8 GB (Ryzen 5 3600XT, 62 GB DDR4), 128K context, q8_0 caches, the model's expert
> profile. Greedy answers of 300 tokens, two runs each. "Code": a 373-token prompt that asks for a 25-line Python
> file back with a function renamed; "text": a 37-token question on hash tables.

| | `--window 1` | `--window 4` | Guesses right |
| --- | ---: | ---: | ---: |
| RTX 2070 Super, IQ4_NL, code | 35.1-35.4 tokens/s | 38.8-40.0 | 102 of 119, 115 of 131 |
| RTX 2070 Super, IQ4_NL, text | 36.7-37.8 | 36.7-37.2 | 2 of 12, 4 of 14 |
| RTX 2070 Super, UD-Q2_K_XL, code | 47.1-47.4 | 47.9-48.1 | 40 of 60, 40 of 60 |
| RTX 2070 Super, UD-Q2_K_XL, text | 48.2-48.4 | 48.8-48.8 | 14 of 19, 13 of 19 |

On a card that holds every expert a window goes through the model as one CUDA graph, as a token alone does (see
above), so the guesses gain on top of the graph's gain. With `--window 4` on a rented RTX 3090 the host handed the card
5.6 launches a window instead of 787 (IQ4_NL) and 24 instead of 921 (UD-Q2_K_XL; nsys, the 128 tokens between a 32-
and a 160-token answer to the 302-token code prompt).

> **Measured on:** a rented RTX 3090 24 GB, every expert of both sizes in VRAM, strata-poly alone, a 16K context,
> q8_0 caches, 256 tokens written, two runs each. "Code": setup's 302- and 1,500-token code prompts; "prose": a
> 302-token prose prompt. Guesses right: `--window 4`'s.

| | `--window 1` | `--window 4` | `--window 8` | Guesses right |
| --- | ---: | ---: | ---: | ---: |
| IQ4_NL, code, 302 | 70.8-70.9 tokens/s | 72.8 | 71.5-71.7 | 31 of 50 |
| IQ4_NL, code, 1,500 | 69.8-69.9 | 73.3 | 71.7-71.8 | 54 of 93 |
| IQ4_NL, prose, 302 | 70.7-70.8 | 69.9-70.0 | 69.9-70.0 | 4 of 9 |
| UD-Q2_K_XL, code, 302 | 84.1 | 89.4-89.5 | 88.3-88.7 | 50 of 95 |
| UD-Q2_K_XL, code, 1,500 | 82.8 | 93.8-93.9 | 91.8-92.5 | 70 of 114 |
| UD-Q2_K_XL, prose, 302 | 83.8-84.0 | 120.8-121.3 | 127.8-128.2 | 123 of 128 |

There `--window 4` and `--window 8` wrote the same 256 tokens as `--window 1`, with the same five likeliest tokens at
every step, and the same with `--no-graphs` (`STRATA_FREEZE_EXPERTS=1`, both sizes, the three prompts). A window of up
to 8 tokens has its experts routed on the card, as a token's are. With the conversation's guesses alone `--window 8`
gains no more than `--window 4` on code.

With every expert on the CPU (`--no-gpu-experts`), `--window 4` wrote the same 200 tokens as `--window 1`, with the
same five likeliest tokens and their probabilities at every step: IQ4_NL and UD-Q2_K_XL, three prompts each, greedy
(RTX 2070 Super, q8_0 caches). `tools/window_agree.py` runs that check on a model and a prompt of your choice; on
the code prompt (both sizes) and on its own repeated prompt (UD-Q2_K_XL) it found the same 200 tokens and five
likeliest at every step, with 35 to 128 guesses right. With
the card's expert cache holding the same experts (`STRATA_FREEZE_EXPERTS=1`, and for `--window 1` a larger
`--vram-reserve-mib`, because a window's rows take VRAM from the cache), UD-Q2_K_XL's answers were the same too, code
and text. Where the two caches held one expert apart (IQ4_NL: 1,006 and 1,007 of 3,072), they parted at a token the
model rated about as likely as another.

Without `--window`, 200 greedy tokens were compared with an earlier build's (the same tokens and twenty likeliest
tokens with their probabilities): IQ4_NL and UD-Q2_K_XL, fp16 and q8_0 caches, code and text prompts, every expert on
the CPU and with the cache holding the same experts. 15 of the 16 runs were identical. In the 16th (UD-Q2_K_XL, fp16,
code, experts cached) the two caches never held the same experts (1,864 and 1,869 at load), and the answers first
differ at a token the model rated about as likely as another (ln p1/p2 0.15).

## Guesses from its DSpark drafter

Unsloth publishes a small drafter for Inkling 2 Mini, [dspark-Inkling-2-Mini-Q8_0.gguf](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
(0.4 GB, three dense layers that read the model's own layers 5, 23, 41 and its output, as llama.cpp's PR 32218 runs
it). With it the engine guesses three tokens a round and puts them through the model with the known one, in one
window. A guess counts only when it is the token the model writes itself. With every expert on the card, or every
expert on the CPU, the greedy answer is the one written without the drafter, token for token. With the experts split
between the card and the CPU (an 8 GB card) it is **not** lossless: the drafter's 0.4 GiB leaves room for fewer
experts in the cache, and the answer can part where the model rates two tokens about equally (see "How it was
checked" below).

**Turning it on.** Setup asks where the card holds every expert of the size beside the drafter (an RTX 3090 24 GB
with either size, a 16 GB card with UD-Q2_K_XL); the answer's default is no. `--dspark yes` sets it up on any card,
`--dspark no` leaves it out. By hand: `"--draft-model", "/path/to/dspark-Inkling-2-Mini-Q8_0.gguf"` in the run
config's `"args"`. A window is then 4 tokens (`--window N` sets it, `--window 1` turns the drafter off), and
`--lookup` still adds the conversation's repeats. Not with `--parallel` (`--batch`) or several cards.

**Where it pays.** On a card that holds every expert, and on text it guesses well: code, text that repeats the
prompt. On prose it guesses about half the tokens right and gains little or nothing. On a card that holds few of the
experts it is no faster or slower: the guessed tokens route to experts of their own, and those run on the CPU. The
model's window of guesses is recorded as one CUDA graph (see above); the drafter's own launches go one by one.

> **Measured on:** a rented RTX 3090 24 GB (every expert of both sizes in VRAM; 256 tokens written, two runs each,
> setup's 302- and 1,500-token code prompts) and a rented RTX 2070 Super 8 GB (31% of IQ4_NL's experts and 68% of
> UD-Q2_K_XL's in VRAM; 200 tokens, one run; "code": the 373-token prompt above, a Python file given back with a
> function renamed; "prose": the 37-token question on hash tables), 2026-11-21/22, strata-poly alone, greedy, q8_0
> caches, the drafter at its window of 4.

| | without | with the drafter | guesses right |
| --- | ---: | ---: | ---: |
| RTX 3090, IQ4_NL, code, 302 | 70.8-70.9 tokens/s | 72.3-72.9 tokens/s | 61 of 83, 66 of 84 |
| RTX 3090, IQ4_NL, code, 1,500 | 69.8-69.9 | 78.1-78.4 | 160 of 214 |
| RTX 3090, UD-Q2_K_XL, code, 302 | 84.1 | 103.5-103.9 | 155 of 216 |
| RTX 3090, UD-Q2_K_XL, code, 1,500 | 82.8 | 100.6-100.8 | 151 of 230 |
| RTX 2070 Super, IQ4_NL, code | 17.8 | 17.1 | 109 of 116 |
| RTX 2070 Super, IQ4_NL, prose | 20.4 | 18.2 | 34 of 59 |
| RTX 2070 Super, UD-Q2_K_XL, code | 28.4 | 31.6 | 97 of 112 |
| RTX 2070 Super, UD-Q2_K_XL, prose | 33.8 | 30.3 | 37 of 44 |

The RTX 2070 Super runs a token launch by launch with or without the drafter (it holds only some of the experts). On
the RTX 3090, prose was measured only while windows still ran launch by launch: there the drafter wrote 2% slower
with IQ4_NL (58.8-59.1 against 60.3 tokens/s, 64 of 120 guesses right) and 9% faster with UD-Q2_K_XL (74.3-74.7
against 68.3-68.7, 114 of 199).

Greedy and frozen, the drafter at windows 4 and 8 wrote the same 256 tokens with the same five likeliest at every step
as `--window 1` and as `--no-graphs` (both sizes, those two prompts and a 302-token prose one).

**VRAM.** The drafter's weights, scratch and window layers' caches took 0.4 GiB at load (the expert cache was that
much smaller with it on the RTX 2070 Super, both sizes), and its full-attention layer keeps 4 KiB a context token in
fp16 whatever `--kv` says (0.5 GiB at 128K). It comes out of the expert cache; setup counts it.

**How it was checked.** Its guesses against llama.cpp's own drafter (PR 32218, 88025eb) on llama.cpp's inputs
(`poly_dspark_parity`, a record of llama.cpp's run, IQ4_NL, seven guesses a round): 84 of 84, 178 of 179 and 137 of
138 guesses the same on a code, a prose and a 1,523-token prompt (RTX 3090), the largest score difference 0.46. With
every expert on the CPU (`--no-gpu-experts`), its answers were the same 200 tokens with the same five likeliest at
every step as without it (both sizes, code and prose, RTX 2070 Super); with every expert on the card the same 200
tokens (both sizes, code and prose, RTX 3090). On an RTX 2070 Super 8 GB (961 of IQ4_NL's 3,072 experts in VRAM
without the drafter, 875 with it; UD-Q2_K_XL 2,056 and 1,881), greedy answers with the drafter parted from the ones
without it in 12 of 14 prompts (both sizes, code and prose, 302 to 8,192 tokens, 256 tokens written): each at a step
where the two likeliest tokens were within 0.016-0.18 nats of each other (ln(p1/p2) in the run without the drafter),
as with `--window` above. The cache holds other experts than without the drafter, so a few scores move a little.

## Long contexts

A written token's attention reads the whole cache of its 8 full-attention layers, about 17 KiB a token of context in
q8_0, so past 100K it takes most of a token's time. With `--kv q8_0`, `STRATA_KV_INT8=1` (the name Kolibri's opt-in
has) reads a full layer's cache by `__dp4a` once its room is 32K or more (from the 16,385th token), with the queries
quantized to Q8_0 codes too (a 32-value block's delta its largest |value| / 127, about as llama.cpp's flash-attention
kernels do for a quantized cache). Window layers, shorter contexts, the prompt and fp16 caches are left as they are:
there the default kernel was as fast or faster. Unlike Kolibri's, it does not touch the prompt. It is off by default:
it changes the arithmetic, so answers can part from the default's at near-ties.

> RTX 2070 Super 8 GB (448 GB/s). One token's attention of one full layer, q8_0 cache (`poly_inkling_kernels_test
> decode`); and UD-Q2_K_XL with `--kv q8_0`, a 128K context, one engine per prompt, 64 tokens written

| | default | `STRATA_KV_INT8=1` |
|---|---|---|
| one layer, room 32K | 268 us | 245 us |
| one layer, room 128K | 1.06 ms | 0.98 ms |
| one layer, room 512K | 4.29 ms | 4.00 ms |
| one layer, room 1M | 8.98 ms (254 GB/s) | 8.32 ms (274 GB/s) |
| writing after a 32,768-token prompt | 39.5 tok/s | 40.9 tok/s |
| writing after a 131,008-token prompt | 29.3 tok/s | 29.9 tok/s |

On this card most of a token's time at 128K goes to the experts that do not fit in its VRAM, so the gain in writing
speed is small.

How far it moves the answers, measured on the same card with the experts held fixed at load
(`STRATA_FREEZE_EXPERTS=1`, the same cache both ways), the default against `STRATA_KV_INT8=1`:
- teacher-forced, a 19,500-token prompt then 64 fed tokens: KL 0.0026 a row on average (largest 0.012), the same
  most likely token at 65 of 65 rows (IQ4_NL); 0.0091 (largest 0.20), 63 of 65 (UD-Q2_K_XL). After a
  30,000-token prompt and 32 fed tokens: 0.0002 (largest 0.003) and 0.0004 (largest 0.008), 33 of 33 both.
- greedy, 256 tokens written after the 19,500- and the 30,000-token prompts: all 256 the same, both files.
- the checks against llama.cpp (below) are prompts of up to 1,500 tokens, where it does not apply; with it applied
  to every layer and room (a build that did, for the test), they stayed in range: 31 or 32 of 32 steps, prompt
  cosine 0.99938 to 0.99971.

## Two or more cards

Several NVIDIA cards split the model's layers as they do for the other models of strata-poly
([MULTI_GPU.md](../MULTI_GPU.md#the-strata-poly-models)): `--gpus 0,1` at setup, or `"gpu": [0, 1]` and
`"layer_split"` in the run config. Each card keeps the routed experts of its own layers in its VRAM, so two cards hold
more of them than one. The twelve layers that run three times (6 to 17) stay on one card: a later pass runs the
experts the first one has in VRAM. So a card starts at layer 1 to 6 or 18 to 22, which the engine says when it is
given another (`"layer_split": "18"` puts layers 0 to 17 on the first card and 18 to 23 with the output on the second).
`"auto"` shares the layers out by each card's free VRAM and then takes 6 or 18, whichever is nearer. What a card hands
the next one is a token's residual streams, the four blocks' starts and the loop's messages among them: 64.5 KB a token.
A card that holds every expert of its own layers runs a token's layers there as one CUDA graph, as a single card that
holds them all does ([How fast is it?](#how-fast-is-it)); the others run theirs launch by launch.

Two RTX 3070 8 GB split at 18 wrote 12% to 21% faster than one of them. Split at 6, both models wrote slower than
at 18 (an earlier build on another box).

> **Measured on:** 2 x RTX 3070 8 GB (rented; PCIe 3.0 x16, no P2P, so the hand-off goes through host memory), a Xeon
> E5-2650 v4 (24 threads), q8_0 caches, the experts each card holds in VRAM, greedy answers of 256 tokens, two runs
> each. The CPU has more cores than a desktop's, so the one-card numbers, and with them what a
> second card adds, are not those of a PC with fewer cores.

| | One card | Split at 18 |
| --- | ---: | ---: |
| IQ4_NL, after 302 tokens | 44.9-45.1 tokens/s | 52.7-52.8 (+17%) |
| IQ4_NL, after 1,500 tokens | 44.9 | 50.2-50.4 (+12%) |
| UD-Q2_K_XL, after 302 tokens | 56.6-56.9 | 68.0-68.2 (+20%) |
| UD-Q2_K_XL, after 1,500 tokens | 54.3-54.5 | 65.6-65.7 (+21%) |

At 18 the IQ4_NL cards held 1,175 of layers 0 to 17's 2,304 experts and 694 of layers 18 to 23's 768; UD-Q2_K_XL's
2,194 and all 768. `"auto"` chose 18 too. The prompts read faster split, except UD-Q2_K_XL's of 1,500 tokens: IQ4_NL
221-226 and 640-650 tokens/s on one card, 257-269 and 989-990 split; UD-Q2_K_XL 363-374 and 1,084-1,093 on one card,
493-503 and 1,089-1,096 split. With every expert on the CPU the split gave the same logits as one card to the byte
(both sizes, fp16 and q8_0, split at 18 and `"auto"`; an earlier build at 6 and 3 too), and with the experts in VRAM
held still the split gave the same logits and greedy answers with and without `--no-graphs`.

## Images

Setup asks whether you want pictures; the answer is no unless you say yes, so `--yes` installs text only.
`--vision cpu` (or `gpu`, or `yes` for setup's choice by VRAM) adds them:

```bash
./setup.sh --family inkling --vision cpu
```

That downloads ggml-org's image encoder, `mmproj-Inkling-2-Mini-Q8_0.gguf` (0.35 GB), and compiles strata-vision
from llama.cpp 29b52d398b, the commit that added the model and its encoder, into `engine-poly/vision-inkling/`. The
other models' encoder (`engine-poly/strata-vision`, an older llama.cpp) is not changed. The encoder is llama.cpp's
own (its `mtmd` library); PolyStrata places its rows where the template writes the picture
(`<|content_image|>` and one `<|unused_200054|>` for each of the picture's rows after it, in a block of its own, as
llama.cpp does) and does not norm them, as llama.cpp does not. A picture in a tool's message goes in the same way.

A picture larger than the token limit (setup's 1,024 on the GPU, 300 on the CPU) is made smaller first, as
llama.cpp does with the same limit. A picture costs its rows in the context: 640 x 640 pixels is 256 tokens.

On an 8 GB card setup puts the encoder on the CPU. Measured on an RTX 2070 Super 8 GB with a Ryzen 5 3600XT
(6 encoder threads), IQ4_NL, flash attention on in the encoder (its default):

| | 256 rows (640 x 640) | 990 rows (1000 x 2000, at the 1,024 limit) | VRAM |
| --- | ---: | ---: | ---: |
| encoder on the CPU | 2.1 s | 12.3 s | none |
| encoder on the GPU (`--vision gpu`) | 0.06 s | 0.31 s | 576 MiB after its warm-up |

The photo below (208 rows) took 1.7 s on the CPU (setup's `--vision cpu` build, 6 threads) and 55 ms on the GPU.

With the encoder on the GPU the engine held 908 routed experts in VRAM instead of 1,038. A prompt with a picture
was read at the speed of a text prompt: 282 tokens with a 256-row picture in 1.2 to 1.35 s (209 to 239 tokens/s in
5 server runs, the encoder on the CPU in 3 of them, on the GPU in 2), 323 tokens of text in 1.4 to 1.5 s (216 and 226
tokens/s, 2 runs). A whole request (the encode, the prompt and an answer of 90 to 170 tokens) took 6.1 to 8.4 s with
the encoder on the CPU and 3.7 to 6.0 s with it on the GPU.

**UD-Q2_K_XL** reads pictures too (the same encoder; `--model UD-Q2_K_XL --vision gpu`). With the encoder on the GPU
it read the 282-token prompt at 354 tokens/s and answered at 50 tokens/s; a whole request took 3.3 to 3.8 s.

**How it was checked.** Against llama.cpp 29b52d398b's `mtmd` on the CPU, for 6 cases (7 drawn pictures from 1 to
990 rows, two of them in one prompt) and a photo (llama.cpp's `tools/mtmd/test-1.jpeg`, a newspaper page, 640 x 488
pixels, 208 rows), with the Q8_0 encoder and IQ4_NL unless named:

- **Rows.** strata-vision's were bit-identical to llama.cpp's with flash attention off (the drawn pictures). With
  flash attention on (the default) their cosine similarity was 0.98 or more, 0.999 on average (the photo: 0.970 or
  more); with the encoder on the GPU 0.965 or more, 0.998 on average, and for the photo 0.957 or more, 0.999 on
  average.
- **Logits.** The engine, given llama.cpp's rows, against llama.cpp: cosine similarity of the prompt's logits 0.99986
  or more, and greedy decoding chose the same token in all 33 positions (the prompt's and 32 steps) for every
  case and the photo. With `--kv q8_0`: 32 of 33 for the 640 x 640 picture (the miss where llama.cpp's top two were
  0.08 apart; a second run with the fp16 cache missed there too), 33 of 33 for the 990-row picture and the photo
  (0.99980). With the photo split across calls (`--prompt-batch 64`): 0.99975, 33 of 33. With the rows normed like
  a token's, as the engine did first, it was 0.98485 or more and 29 to 32 of 33. Rows from strata-vision with flash
  attention on, from the GPU encoder, or from the BF16 encoder gave the same agreement (0.99989 or more, 33 of 33,
  on 2 pictures).
- **UD-Q2_K_XL.** The engine on Unsloth's file against llama.cpp on the same weights (converted to the form
  llama.cpp loads, as for the text checks): the photo 0.99990 (33 of 33; `--kv q8_0` 0.99985, 33 of 33), the 640 x
  640 picture 0.99993 (32 of 33, at step 11), the two pictures 0.99984 (33 of 33).
- **Answers.** Asked "Describe the picture in one sentence." at temperature 0 about a 640 x 640 test picture, the
  server and `llama-mtmd-cli` both answered: *A light blue background shows a red circle in the upper left, the word
  "HELLO" in bold black letters across the middle, and a green square in the lower right.* Two server runs with the
  encoder on the CPU gave that answer, one of them installed by setup with `--vision cpu`; two others and two with
  it on the GPU worded the same picture a little differently. The engine's runs are not bit-for-bit repeatable, and
  the reasoning before the answer has a near tie. About the photo, the `--vision cpu` install answered word for
  word as `llama-mtmd-cli`: *A front page of The New York Times from July 20, 1969, with the bold headline "MEN WALK
  ON MOON" and a black-and-white photo of the lunar surface and spacecraft.* The `--vision gpu` install read the
  same headline, subhead and date and answered *A vintage New York Times front page from July 20, 1969, with the
  bold headline "MEN WALK ON MOON" and a black-and-white photo of the lunar surface.* With UD-Q2_K_XL (encoder on
  the GPU) the server and `llama-mtmd-cli` both answered *A vintage New York Times front page headlined "MEN WALK ON
  MOON" with a black-and-white photo of the lunar surface and spacecraft.*; about the 640 x 640 picture both named
  the red circle, "HELLO" and the green square, in other words.

Why these settings: the Q8_0 encoder is half the BF16 one's download, uses 576 instead of 904 MiB of VRAM and is
faster (256 rows: 2.1 against 3.3 s on the CPU), and the engine's logits were as close to llama.cpp's with either.
Flash attention off made the encoder slower (256 rows 2.8 s, 990 rows 22.7 s on the CPU) and used 1,584 MiB of VRAM
on the GPU, for no change in the logits that could be measured (on the drawn pictures).

## How it talks

The server reads these from the tokenizer that setup writes from the GGUF (`pre` = `inkling2`):

- **Tokens.** Byte-level BPE with 200,058 tokens, llama.cpp's `inkling2` split (its GPT-4o pattern) and, as in the
  model's `tokenizer.json`, `ignore_merges`: a piece that is a token as a whole is that token. No token starts a
  prompt. A reply ends at `<|content_model_end_sampling|>` (the GGUF's end of text) or at `<|endoftext|>`.
- **Reply.** A reply is a list of blocks. Each starts with `<|message_model|>` and ends with `<|end_message|>`. A
  `<|content_thinking|>` block is the reasoning, a `<|content_text|>` block is the answer, and a tool call is
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`. This is the form
  llama.cpp reads (its Inkling parser, ggml-org/llama.cpp PR 32381). The model writes the first header itself.
- **Reasoning effort.** The template writes "Thinking effort level" in a system block. `reasoning_effort` low, medium
  and high give 0.2, 0.7 and 0.9. High (0.9) is the default. Thinking turned off gives 0. The model can still think a
  little at 0: its model card says the lowest effort does not turn reasoning off. "xhigh" and "max" are 0.9 too,
  and "minimal" is thinking off (0), as for every model here: the template's 0.1 and 0.99 are not offered.
- **Sampling.** Setup writes the GGUF's own defaults into the run config: temperature 1.0, top_k 50, top_p 1.0. A
  `"sampling"` block you wrote yourself is kept.
- **Forced calls.** `tool_choice` naming one tool is written for the model after its thinking, or at the start of
  the reply with thinking off. The reply ends with that call: a block the model opens after it is not sent.
  `"required"` with several tools is left to the model.

## How it was checked

Token ids against Hugging Face's `tokenizers` library on the model's `tokenizer.json`, the reference. Prompts and
replies against llama.cpp 29b52d398b, the commit that added the model, built on the CPU and run without the model's
weights. 51 of the strings (53 cases), the 14 prompts and the 11 replies are kept in `serve/fixtures/` and checked by
`serve/test_inkling.py`.

- **Tokens.** 1,006 strings, about 587,000 tokens: English, Japanese, Chinese, Korean, Arabic, Hindi and Russian text,
  code, emoji, whitespace runs, control tokens, PolyStrata's docs and source in pieces, and 300 random strings. Every
  id was the same as Hugging Face's. llama.cpp's differed in 11 of the strings (see below).
- **Prompts.** 14 requests rendered by llama.cpp's chat code and by the server were the same text: effort levels, a
  system prompt, tools, two tool calls and their answers, earlier reasoning, Unicode and quotes.
- **Replies.** 11 replies (llama.cpp's own Inkling tests and a few more) gave the same reasoning, answer and tool calls,
  fed whole, in 5-character pieces and one character at a time.

The engine's output was checked against llama.cpp's on the same files, IQ4_NL and UD-Q2_K_XL, for a short prompt and prompts
of 302 and 1,500 tokens, each run twice: the logits' cosine similarity was 0.9989 to 0.9998, and 16 to 20 of the 20
most likely tokens were the same. With llama.cpp's own 32-token answer fed in one token at a time, the most likely
token was the same at 29 to 32 of the 32 steps. Where it differed, llama.cpp's two best tokens were within 0.14 of each
other in logit, except at one step (UD-Q2_K_XL, the 1,500-token prompt, in every run), where they were 1.0 apart. Two
runs of the same input do not give identical logits here (cosine 0.9992 between two runs of the 302-token prompt).

To compare two builds byte for byte with the experts on the card, give both the same `STRATA_GPU_EXPERTS=N` and the
same copy of the expert profile: each build's cache holds the same N experts for the whole run, frozen, whatever VRAM
the build leaves free. Without it the two builds' caches often differ by a few experts (a build that records more
launches leaves less VRAM), and so do their logits. Pick an N a little below what the smaller build caches on its own
(`poly_logits_dump`'s `N of 3072 experts cached` line, `strata-poly`'s `N of 3072 routed experts in VRAM`); one that
does not fit stops the load with an error. With either knob set the load also logs `cache set N experts fnv <hash>`,
which experts the cache holds: equal counts are not equal sets (UD-Q2_K_XL's experts come in seven sizes and the cache
takes smaller ones past one that does not fit: 1,640 of them can be any of three sets within 2.5 MiB of cache), so
compare the hash, not the count.

## Known differences

- llama.cpp 29b52d398b does not set `ignore_merges` for `inkling2`, so a word that is one token but that no chain of
  merges reaches comes out as several there: `kwargs` is one token here and with Hugging Face, three in llama.cpp.
  That changed 11 of the 1,006 strings above. PolyStrata follows Hugging Face. `serve/fixtures/inkling2_golden.json`
  keeps llama.cpp's ids for one of them (`kwargs`).
- A message whose content is a list of text parts is one block here and one block a part in llama.cpp. The server
  joins the parts for every model.
- llama.cpp checks a tool call against the declared tools with a grammar while the model writes it. PolyStrata has no
  grammar: it reads the call when its block is closed. A call to a tool that was not declared is shown as text.

## How does it work?

The engine side is in `src/poly/inkling.cpp`. The server side is `InklingParser` in `serve/frontend.py`, and the tests
are `serve/test_inkling.py` and `tools/test_setup_inkling.py`.
