# NVIDIA Nemotron Model Reasoning Challenge — 174th place solution

This repository contains my solution for the Kaggle **NVIDIA Nemotron Model Reasoning Challenge**. The final submission achieved:

| Metric | Score |
|---|---:|
| Private LB | **0.860** |
| Public LB | **0.852** |
| Final place | **174 / 4182 teams** |

The solution is based on supervised fine-tuning of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` with LoRA. During the competition I first focused on improving reasoning chains and solvers, then shifted most of the effort to making the training pipeline stable, memory-efficient, and able to train on long sequences up to **8192 tokens**.

## High-level idea

The main improvement came from changing how the model was trained on chain-of-thought data.

Training only on full reasoning traces makes the model spend most of its loss budget on reproducing the format of a long answer. It can learn how a solution should look, but it does not necessarily learn the small local operations that make the reasoning correct. For harder tasks, especially where the useful signal is buried inside a long trace, the loss becomes too diffuse.

The final approach mixes two types of examples:

1. **Full-task examples** — complete prompts with full reasoning chains and final answers.
2. **Subtask / dense examples** — smaller local reasoning blocks extracted from the original reasoning process. Each block asks the model to solve one local operation inside the larger task.

This made the training signal more targeted: instead of only teaching the model to imitate complete CoT format, the subtasks teach local transformations and intermediate operations directly. The full examples still keep the final answer format and global task structure aligned.

## What worked

### 1. Solver-generated CoT data

I generated reasoning traces with task-specific solvers and filtered examples where the generated answer did not match the ground truth. This gave cleaner data than my earliest baseline with external CoT labels, where many traces did not even mention the true answer.

The pipeline includes solvers for several task families:

- bit manipulation
- encryption
- numeral equations / equation transformations
- unit conversion
- gravitational tasks
- conversion to another numeral system

### 2. Upsampling hard tasks

Early experiments showed that easier tasks dominated training. I added task-level weighting / upsampling so that harder families contributed more often to the loss. This was especially useful for equation transformations, bit manipulation, and encryption.

### 3. Long-context LoRA training

A large part of the competition was spent on training optimization. The final pipeline could train LoRA on sequences up to **8192 tokens** with remaining memory headroom.

Final important training settings:

```yaml
model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
max_seq_len: 8192
lora_r: 32
lora_alpha: 64
lr: 2e-4
batch_size: 4
grad_accum: 8
epochs: 4
loss_impl: cce
```

### 4. Targeted improvements for bit manipulation and numeral equations

The final versions put additional emphasis on bit manipulation and numeral equations. For bit manipulation, I added support for extra operations such as `majority` and `choice`, which later helped on the leaderboard.

## What did not work well

More than half of my competition time was spent trying to make the model solve **cryptarithm** tasks. I tried multiple CoT formats, solver improvements, synthetic examples, and local subtasks, but the model mostly learned the expected format rather than the required operations.

The core problem was that cryptarithm reasoning required many local constraints and intermediate operations, while the traces still had to fit into the token budget. Compressing those operations made the examples easier to fit into context, but also removed the very signal the model needed to learn from. In the end, cryptarithm remained the main unsolved direction in this solution.

## Pipeline

The repository uses DVC stages for the main training pipeline:

```bash
# 1. Generate solver CoTs
python -m src.generate_cot \
  --train_path data/raw/train.csv \
  --output_path data/solved/train-cot.csv

# 2. Prepare training data with synthetic / subtask examples
python -m src.prepare_dataset \
  --data_path data/solved/train-cot.csv \
  --output_path data/generated/train_with_syntetic.csv \
  --seed 31231 \
  --tokenizer_path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

# 3. Train LoRA
python -m src.sft_train \
  --model_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --data algorithmic_cot \
  --train_path data/generated/train_with_syntetic.csv \
  --output_dir models/sft_v0.0.29 \
  --batch_size 4 \
  --grad_accum 8 \
  --epochs 4 \
  --lr 2e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --max_seq_len 8192 \
  --loss_impl cce \
  --cce_impl cce
```

Or run the pipeline through DVC:

```bash
dvc repro
```

## Reproducing

```bash
git clone https://github.com/bezbahen0/nvidia-nemotron-model-reasoning-challenge.git
cd nvidia-nemotron-model-reasoning-challenge

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data/raw
# Put Kaggle train.csv into data/raw/train.csv

dvc repro
```

Notes:

- Competition data is expected at `data/raw/train.csv`.
- Training is GPU-heavy. The final configuration was optimized for long sequences and a 30B MoE model.
- W&B logging is supported by the training script.

## Repository structure

```text
.
├── data/                  # raw, solved, generated, prepared and score files
├── notebooks/             # EDA and telemetry notebooks
├── src/
│   ├── augmentation/      # synthetic / subtask / upsampling logic
│   ├── solvers/           # task-specific solvers and CoT generators
│   ├── generate_cot.py    # solver CoT generation
│   ├── prepare_dataset.py # dataset construction
│   ├── sft_train.py       # LoRA SFT training
│   └── metric.py          # local metric / answer extraction utilities
├── dvc.yaml               # DVC pipeline
├── params.yaml            # training and data parameters
└── requirements.txt       # dependencies
```

## Experiment log, condensed

### Baseline

The first baseline used external Gemini-generated CoT labels. It was useful as a starting point, but the data quality was not good enough: many traces did not contain the true answer. This pushed me toward generating my own solver-based reasoning chains.

### Solver and CoT iterations

I iterated on solvers and CoT formats for bit manipulation, encryption, numeral equations, unit conversion, gravitational tasks, and numeral-system conversion. A stable intermediate version reached strong solver-generated coverage and became the basis for later SFT experiments.

### Hard-task weighting

I added upsampling for difficult task families so that the model would see more of the examples it struggled with instead of mostly optimizing on easy cases.

### Dense / subtask training

After studying Tong Huikang's public Nemotron solution, I moved from “train only on complete CoTs” toward “train on both complete CoTs and local subtasks.” This became the key idea of the final solution.

### Final long run

Most experiments were trained for one epoch to save time and evaluate quickly. After observing the learning dynamics, I ran a longer final training with 4 epochs. That gave the most stable final model and the best leaderboard result.

## Acknowledgements

Thanks to:

- **Tong Huikang** for the public Nemotron solution and writeup, which inspired the subtask/dense-training direction.
- **NVIDIA** and **Kaggle** for organizing the competition.
- The Kaggle community for shared baselines, notebooks, and discussions.
