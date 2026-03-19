import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import os

# Setup
OPTIONS = ["R", "G", "B"]
idx = {o: i for i, o in enumerate(OPTIONS)}

# Directories
RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Population definitions
TYPE_1 = ["R", "G", "B"]
TYPE_2 = ["G", "B", "R"]
TYPE_3 = ["B", "R", "G"]

def sample_voter(distribution):
    types = [TYPE_1, TYPE_2, TYPE_3]
    return random.choices(types, weights=distribution)[0]

def prefers(ranking, a, b):
    return ranking.index(a) < ranking.index(b)

# Data generation
def generate_dataset(n_samples, distribution):
    data = []
    for _ in range(n_samples):
        a, b = random.sample(OPTIONS, 2)
        voter = sample_voter(distribution)
        if prefers(voter, a, b):
            data.append((a, b))
        else:
            data.append((b, a))
    return data

# RLHF (probabilistic)
def rlhf_scores(data):
    wins = defaultdict(int)
    for winner, _ in data:
        wins[winner] += 1
    total = sum(wins.values())
    return {o: wins[o] / total for o in OPTIONS}

# RLHF (argmax version)
def rlhf_argmax(probs):
    winner = max(probs, key=probs.get)
    return {o: 1.0 if o == winner else 0.0 for o in OPTIONS}

# Margin matrix
def build_margin_matrix(data):
    M = np.zeros((len(OPTIONS), len(OPTIONS)))
    for a, b in data:
        i, j = idx[a], idx[b]
        M[i][j] += 1
        M[j][i] -= 1
    return M

# Maximal Lottery (LP)
def maximal_lottery(M):
    n = len(OPTIONS)
    c = [0]*n + [-1]
    A_ub = []
    b_ub = []
    for j in range(n):
        constraint = [-M[i][j] for i in range(n)]
        constraint.append(1)
        A_ub.append(constraint)
        b_ub.append(0)
    A_eq = [[1]*n + [0]]
    b_eq = [1]
    bounds = [(0, 1)]*n + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if not res.success:
        raise ValueError("LP failed")
    p = res.x[:n]
    return {OPTIONS[i]: float(p[i]) for i in range(n)}

# Plots
def plot_results(rlhf_prob, rlhf_det, ml, title, filename):
    labels = OPTIONS
    x = np.arange(len(labels))
    width = 0.25
    rlhf_vals = [rlhf_prob[o] for o in labels]
    rlhf_det_vals = [rlhf_det[o] for o in labels]
    ml_vals = [ml[o] for o in labels]

    plt.figure()
    plt.bar(x - width, rlhf_vals, width, label="RLHF (prob)")
    plt.bar(x, rlhf_det_vals, width, label="RLHF (argmax)")
    plt.bar(x + width, ml_vals, width, label="Max Lottery")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Saved plot to: {filepath}")
    plt.close()

# Experiment runner
def run_experiment(distribution, label, filename):
    data = generate_dataset(2000, distribution)
    rlhf_prob = rlhf_scores(data)
    rlhf_det = rlhf_argmax(rlhf_prob)
    M = build_margin_matrix(data)
    ml = maximal_lottery(M)

    print(f"\n--- {label} ---")
    print("RLHF (prob):", rlhf_prob)
    print("RLHF (argmax):", rlhf_det)
    print("Maximal Lottery:", ml)

    plot_results(rlhf_prob, rlhf_det, ml, label, filename)

# Main
if __name__ == "__main__":
    # Perfect cycle
    run_experiment(
        distribution=[1/3, 1/3, 1/3],
        label="Perfect Cycle",
        filename="perfect_cycle.png"
    )
    # Slight perturbation
    run_experiment(
        distribution=[0.4, 0.3, 0.3],
        label="Perturbed Cycle (0.4, 0.3, 0.3)",
        filename="perturbed_cycle.png"
    )
    # Strong perturbation
    run_experiment(
        distribution=[0.5, 0.25, 0.25],
        label="Strong Perturbation",
        filename="strong_perturbation.png"
    )
