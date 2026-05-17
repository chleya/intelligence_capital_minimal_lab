import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_mechanism_comparison(results_dict, output_path="results/mechanism_comparison.png"):
    names = list(results_dict.keys())
    matches = [results_dict[n].get("best_action_match", 0) for n in names]
    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), matches, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Best Action Match")
    ax.set_title("Throttling Mechanism Comparison")
    ax.axhline(y=0.33, color="gray", linestyle="--", label="Random (33%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


def plot_death_checks(death_results, output_path="results/death_checks.png"):
    names = list(death_results.keys())
    triggered = [sum(1 for d in death_results[n] if d.get("triggered")) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), triggered, color=["green" if t == 0 else "red" if t >= 4 else "orange" for t in triggered])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Death Conditions Triggered")
    ax.set_title("Death Condition Audit per Mechanism")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


def plot_bad_debt(audit_results, output_path="results/bad_debt.png"):
    names = list(audit_results.keys())
    bdr = [audit_results[n]["bad_debt_ratio"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), bdr, color=["green" if b < 0.5 else "red" for b in bdr])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Bad Debt Ratio")
    ax.axhline(y=0.5, color="red", linestyle="--", label="Death threshold (50%)")
    ax.set_title("Bad Debt Ratio per Mechanism")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()


def plot_intelligence_appreciation(mechanism_names, iar_values, output_path="results/iar.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(mechanism_names)), iar_values,
                  color=["green" if v > 0 else "red" for v in iar_values])
    ax.set_xticks(range(len(mechanism_names)))
    ax.set_xticklabels(mechanism_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Intelligence Appreciation Rate (IAR)")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title("IAR per Throttling Mechanism")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()