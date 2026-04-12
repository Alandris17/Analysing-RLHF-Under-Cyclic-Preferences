# Analysing RLHF Under Cyclic Preferences

This project investigates how **Reinforcement Learning from Human Feedback (RLHF)** behaves under **cyclic preferences**, and compares it with the **Maximal Lottery** solution from social choice theory.

It extends ideas from:
> *"Jackpot! Alignment as a Maximal Lottery"* — Maura-Rivero et al.

---

## Overview

In many real-world settings, preferences are not transitive. Instead, they can form cycles:

- R > G  
- G > B  
- B > R  

This leads to a **Condorcet cycle**, where no single option is universally preferred.

This project simulates such scenarios and compares three approaches:

1. **RLHF (probabilistic)**  
   - Learns a probability distribution from pairwise comparisons

2. **RLHF (argmax / deterministic)**  
   - Picks the single most likely option

3. **Maximal Lottery**  
   - A game-theoretic solution that returns a mixed strategy robust to cycles

---

## Key Insight

- **RLHF (argmax)** can behave poorly in cyclic settings, collapsing to a single arbitrary winner  
- **RLHF (probabilistic)** smooths this, but still reflects sampling bias  
- **Maximal Lottery** provides a principled equilibrium over options

---

## Experiments

We simulate three preference regimes:

### 1. Perfect Cycle
Uniform distribution over cyclic voters:
- (1/3, 1/3, 1/3)
- All methods produce near-uniform distributions
- RLHF (argmax) becomes unstable due to ties

---

### 2. Perturbed Cycle
Slight bias introduced:
- (0.4, 0.3, 0.3)
- RLHF (argmax) collapses to a single winner
- Maximal Lottery still distributes probability

---

### 3. Strong Perturbation
Stronger bias:
- (0.5, 0.25, 0.25)
- RLHF becomes increasingly skewed
- Maximal Lottery remains more balanced

---

## Example Outputs

Plots are automatically generated and saved in the `Results/` directory:

- `perfect_cycle.png`
- `perturbed_cycle.png`
- `strong_perturbation.png`

Each plot compares:
- RLHF (prob)
- RLHF (argmax)
- Maximal Lottery

---

## How It Works

### Data Generation
- Voters are sampled from predefined cyclic preference types
- Pairwise comparisons are generated randomly

---

### RLHF
- Counts wins from comparisons
- Normalises into probabilities

---

### Maximal Lottery
- Constructs a **margin matrix**
- Solves a **linear program** to find a mixed equilibrium

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

```bash
python main.py
```

This will:
- Generate datasets (1,000,000 samples per experiment)
- Compute all methods
- Save plots to `Results/`

---

## Project Structure

```bash
.
├── main.py              # Core implementation
├── README.md            # Project documentation
├── requirements.txt     # Dependencies
└── Results/             # Generated plots
```

---

## Technical Details

### Margin Matrix

Encodes pairwise preference differences:
- Positive = row option beats column option
- Negative = losses

---

### Linear Program (Maximal Lottery)

Maximises worst-case payoff:
```bash
max v
subject to:
    transpose(p) M >= v
    sum(p) = 1
    p >= 0
```

Solved using `scipy.optimize.linprog`.

---

## Limitations

- Synthetic preference distributions only
- Only 3 options (R, G, B)
- Large sample size required for stability
- RLHF implementation is simplified (count-based)

---

## Future Work

- Extend to more than 3 options
- Use real human preference datasets
- Compare with other voting rules (e.g. Borda, Copeland)
- Integrate into actual RLHF pipelines

---

## References 

- Maura-Rivero et al., Jackpot! Alignment as a Maximal Lottery

---