# autoresearch — parameter-golf

This is an autonomous research loop for the [Parameter Golf Challenge](https://github.com/openai/parameter-golf).
Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Goal

Train the best language model that:
- Fits in a **16 MB artifact** (code + compressed int8+zlib model ≤ 16,000,000 bytes)
- Trains in under **10 minutes on 8×H100 GPUs**
- Achieves the **lowest val_bpb** (bits-per-byte) on the FineWeb validation set

## Available hardware

| Platform | Hardware | Script | Use case |
|----------|----------|--------|----------|
| **MacBook Pro M4** | 16GB unified memory | `train_gpt_mlx.py` | Fast local iteration, architecture experiments |
| **Google Colab** | T4 (free) or A100 (Pro) | `train_gpt.py` | Medium iteration with CUDA, single-GPU |
| **RunPod** ($25 budget) | H100 (single or multi) | `train_gpt.py` | Final validation runs, challenge-realistic timing |

**Strategy**: Iterate cheaply on M4/Colab, validate winners on RunPod.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current branch.
3. **Read the in-scope files**: The key files are:
   - `README.md` — challenge overview, leaderboard, submission format.
   - `train_gpt.py` — **the PyTorch file you modify** (for CUDA GPUs).
   - `train_gpt_mlx.py` — **the MLX file you modify** (for Apple Silicon). Keep changes in sync.
   - `data/cached_challenge_fineweb.py` — data download script (read-only).
   - `records/` — existing leaderboard submissions for inspiration.
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains training shards and `./data/tokenizers/fineweb_1024_bpe.model` exists. If not, tell the human to run `python data/cached_challenge_fineweb.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

### MacBook Pro M4 (16GB) — Local development

The MLX script runs natively on Apple Silicon. To stay within 16GB:

```bash
# Quick 2-minute experiment (recommended for iteration)
MAX_WALLCLOCK_SECONDS=120 ITERATIONS=5000 \
  TRAIN_BATCH_TOKENS=65536 GRAD_ACCUM_STEPS=8 \
  python train_gpt_mlx.py > run.log 2>&1

# Full 10-minute run
python train_gpt_mlx.py > run.log 2>&1
```

Key env vars for memory management on 16GB M4:
- `TRAIN_BATCH_TOKENS=65536` — smaller batch to fit in memory
- `GRAD_ACCUM_STEPS=8` — accumulate to compensate for small batches
- `MLX_EAGER_EVAL=1` — (default) forces evaluation per sub-batch, keeps memory low
- `MLX_MAX_MICROBATCH_TOKENS=4096` — reduce if still OOM

**Note**: val_bpb on M4 won't match H100 exactly (different training speed = different number of steps in time budget), but the *relative ordering* of experiments should be consistent. Use M4 for quick A/B comparisons.

### Google Colab — Single GPU (T4 or A100)

```bash
# Install deps
pip install torch numpy sentencepiece huggingface-hub

# Download data (first time only)
python data/cached_challenge_fineweb.py --train-shards 10

# Quick experiment (2 min)
MAX_WALLCLOCK_SECONDS=120 python train_gpt.py > run.log 2>&1

# Full run
python train_gpt.py > run.log 2>&1
```

On T4 (16GB VRAM), you may need to reduce batch size:
- `TRAIN_BATCH_TOKENS=131072` (default is 524288)

### RunPod — H100 validation runs ($25 budget)

Use RunPod for final validation of promising experiments. At ~$3/hr for 8×H100:
- Budget allows ~8 full 10-minute runs
- Use wisely — only validate your best M4/Colab experiments here

```bash
# Single H100
python train_gpt.py > run.log 2>&1

# Multi-GPU (match challenge setup)
torchrun --nproc_per_node=8 train_gpt.py > run.log 2>&1
```

## What you CAN and CANNOT do

**CAN do:**
- Modify `train_gpt.py` and/or `train_gpt_mlx.py` — everything is fair game: model architecture, optimizer, hyperparameters, training loop, quantization strategy, batch size, model dimensions, number of layers, etc.

**CANNOT do:**
- Modify the data pipeline or tokenizer files in `data/`.
- Install new packages not already available.
- Modify the evaluation logic (`eval_val` function and its helpers). The BPB calculation is the ground truth metric.
- Change the serialization roundtrip validation — the final reported `val_bpb` must be from the int8+zlib roundtripped model.

**The goal is simple: get the lowest val_bpb.**

**Size constraint**: Total artifact (code + compressed model) ≤ 16,000,000 bytes. The script reports this at the end as `Total submission size int8+zlib`.

**Simplicity criterion**: All else being equal, simpler is better.

**The first run**: Always establish the baseline first by running the training script as-is.

## Output format

The training script logs val_bpb at validation checkpoints and at the end:

```
step:20000/20000 val_loss:X.XXXX val_bpb:X.XXXX train_time:XXXXXXms
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX
Total submission size int8+zlib: XXXXXXX bytes
```

Extract the key metrics:
```bash
grep "final_int8_zlib_roundtrip_exact\|Total submission size" run.log
```

For MLX runs (no int8 roundtrip by default), look for the last val_bpb line:
```bash
grep "val_bpb" run.log | tail -1
```

The **final_int8_zlib_roundtrip val_bpb** is the definitive metric for submissions.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_bpb	artifact_bytes	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved — use 0.000000 for crashes
3. total artifact bytes (int8+zlib model + code) — use 0 for crashes or MLX-only runs
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	artifact_bytes	status	description
a1b2c3d	1.1850	4200000	keep	baseline (9 layers 512dim)
b2c3d4e	1.1780	4200000	keep	increase matrix_lr to 0.06
c3d4e5f	1.1900	4200000	discard	switch to GeLU activation
d4e5f6g	0.000000	0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar22`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune the training script with an experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: redirect all output to `run.log`.
   - M4: `MAX_WALLCLOCK_SECONDS=120 TRAIN_BATCH_TOKENS=65536 python train_gpt_mlx.py > run.log 2>&1`
   - Colab: `MAX_WALLCLOCK_SECONDS=120 python train_gpt.py > run.log 2>&1`
   - RunPod full: `torchrun --nproc_per_node=8 train_gpt.py > run.log 2>&1`
5. Read out the results: `grep "val_bpb\|Total submission size" run.log | tail -3`
6. If empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv (do not commit results.tsv — leave it untracked by git).
8. If val_bpb improved (lower), keep the commit.
9. If val_bpb is equal or worse, `git reset --hard HEAD~1` to revert.

## Ideas to explore

Based on the current leaderboard (see `records/track_10min_16mb/`), promising directions include:

- **Model scaling**: Try different depth/width ratios (more layers at narrower width, or fewer layers at wider width)
- **Learning rates**: Tune embed_lr, matrix_lr, scalar_lr, head_lr independently
- **Optimizer**: Muon momentum, backend steps, warmup schedule
- **Architecture**: GQA ratios, MLP expansion factor, different activation functions
- **Quantization**: Improve int8 quantization (per-channel vs per-tensor, clip percentiles)
- **Training dynamics**: Batch size, sequence length, warmdown schedule, gradient clipping
- **Advanced techniques from leaderboard**: Mixed-precision quantization (int5/int6), BigramHash embeddings, value residuals, sliding window attention, stochastic weight averaging

**Timeout**: Each experiment should take at most ~10 minutes for full runs, or ~2 minutes for quick iteration. If a run exceeds 15 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: fix simple issues and re-run, or skip broken ideas.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — read the leaderboard submissions in `records/`, re-read the training script for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.
