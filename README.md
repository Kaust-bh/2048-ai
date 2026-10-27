# 2048 DQN AI

A from-scratch 2048 environment, Deep Q-Network agent, training pipeline, evaluation script, and Pygame visualizer.

## 1. Setup

Python 3.10+ recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On an Apple Silicon Mac, install the PyTorch build supported by your Python version. PyTorch can use Apple's MPS backend in some workloads, but this starter chooses CUDA if available and otherwise CPU. CPU is fine for the first version.

## 2. Test the game rules

```bash
python -m pytest -q
```

Install pytest if needed: `pip install pytest`.

## 3. Train

Start small to confirm everything works:

```bash
python train.py --episodes 100
```

Then try a longer run:

```bash
python train.py --episodes 5000
```

Checkpoints and `training.csv` go into `checkpoints/`. Continue a run with:

```bash
python train.py --episodes 2000 --resume checkpoints/dqn_2048.pt
```

`--episodes` means additional episodes for that run, not the desired total episode number.

## 4. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/dqn_2048.pt --games 100
```

Evaluation turns exploration off. Results are written to `evaluation.csv`.

## 5. Watch the model

```bash
python play.py --checkpoint checkpoints/dqn_2048.pt
```

Controls: SPACE pause, +/- change speed, R restart, ESC quit.

## 6. Important caveats

- DQN is an educational baseline, not the strongest known approach to 2048.
- The reward function is a starting point. Compare reward designs instead of assuming one is optimal.
- The starter DQN does not mask invalid actions in its Bellman target, only when selecting actions. This is simple and runnable, but a next improvement is to use Double DQN and valid-action masking in target computation.
- Training is stochastic. A short run may not reach 2048. Evaluate many games and compare to a random baseline.
- Use fixed evaluation seeds and the same number of games when comparing checkpoints.
