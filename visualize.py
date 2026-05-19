"""
Generates all plots into ./plots/
Run AFTER train.py (and optionally qlearning.py).
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.tree import plot_tree

warnings.filterwarnings("ignore")

DATASET_PATH = "dataset/games.csv"
MODELS_DIR   = "models"
PLOTS_DIR    = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_models():
    with open(f"{MODELS_DIR}/decision_tree.pkl", "rb") as f:
        dt = pickle.load(f)
    mlp_bundle = pickle.load(open(f"{MODELS_DIR}/mlp.pkl", "rb"))
    return dt, mlp_bundle


def save(name):
    path = f"{PLOTS_DIR}/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot 1: Win Rate Bar Chart ────────────────────────────────────────────────
def plot_win_rates(df):
    rates = {}
    for strategy in ["random", "naive", "basic"]:
        sub = df[df["strategy"] == strategy]
        wins = sub[sub["result"].isin(["win", "blackjack"])].shape[0]
        rates[strategy.capitalize()] = wins / len(sub) * 100
    basic_sub = df[df["strategy"] == "basic"]
    ai_wins = basic_sub[basic_sub["result"].isin(["win", "blackjack"])].shape[0]
    rates["AI Model"] = ai_wins / len(basic_sub) * 100

    labels = list(rates.keys())
    values = list(rates.values())
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 60)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Win Rate by Strategy", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=43, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    save("win_rates")


# ── Plot 2: Decision Tree Diagram ─────────────────────────────────────────────
def plot_decision_tree(dt):
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(dt, feature_names=["player_total", "dealer_showing", "usable_ace"],
              class_names=["stand", "hit"], filled=True, rounded=True,
              fontsize=9, ax=ax, max_depth=3)
    ax.set_title("Decision Tree (max_depth=5, top 3 levels shown)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save("decision_tree")


# ── Plot 3: Feature Importance ────────────────────────────────────────────────
def plot_feature_importance(dt):
    features = ["player_total", "dealer_showing", "usable_ace"]
    importances = dt.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([features[i] for i in idx], importances[idx],
                  color=["#3498db", "#e74c3c", "#2ecc71"], edgecolor="white")
    for bar, val in zip(bars, importances[idx]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Importance", fontsize=12)
    ax.set_title("Decision Tree Feature Importance", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(importances) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save("feature_importance")


# ── Plot 4: Heatmap Basic Strategy vs DT ──────────────────────────────────────
def plot_heatmap(dt, df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax_idx, (title, use_model) in enumerate([
        ("Basic Strategy (dataset)", False),
        ("AI Model (Decision Tree)", True),
    ]):
        ax = axes[ax_idx]
        player_totals   = range(8, 22)
        dealer_showings = range(2, 12)
        grid = np.zeros((len(player_totals), len(dealer_showings)))
        for i, pt in enumerate(player_totals):
            for j, ds in enumerate(dealer_showings):
                if use_model:
                    feat = pd.DataFrame([[pt, ds, 0]], columns=["player_total", "dealer_showing", "usable_ace"])
                    grid[i, j] = dt.predict(feat)[0]
                else:
                    sub = df[(df["strategy"] == "basic") & (df["player_total"] == pt) & (df["dealer_showing"] == ds)]
                    if len(sub) > 0:
                        sub = sub.dropna(subset=["actions"])
                        fa = sub["actions"].apply(lambda x: str(x).split(",")[0])
                        grid[i, j] = (fa == "hit").mean() if len(fa) > 0 else 0.5
                    else:
                        grid[i, j] = 0.5
        cmap = mcolors.LinearSegmentedColormap.from_list("bj", ["#2ecc71", "#e74c3c"])
        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(dealer_showings)))
        ax.set_xticklabels([str(d) if d < 11 else "A" for d in dealer_showings])
        ax.set_yticks(range(len(player_totals)))
        ax.set_yticklabels(list(player_totals))
        ax.set_xlabel("Dealer Showing", fontsize=11)
        ax.set_ylabel("Player Total", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
    fig.suptitle("Hit (red) vs Stand (green) Decision Map", fontsize=14, fontweight="bold", y=1.01)
    plt.colorbar(im, ax=axes, label="Hit probability")
    plt.tight_layout()
    save("heatmap")


# ── Plot 5: max_depth Experiment ──────────────────────────────────────────────
def plot_depth_experiment():
    path = f"{MODELS_DIR}/depth_experiment.csv"
    if not os.path.exists(path):
        print("depth_experiment.csv not found, skipping")
        return
    depth_df = pd.read_csv(path, index_col="depth")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(depth_df))
    ax.plot(x, depth_df["train_acc"] * 100, "o-", color="#3498db", label="Train", linewidth=2)
    ax.plot(x, depth_df["test_acc"] * 100,  "s--", color="#e74c3c", label="Test",  linewidth=2)
    for xi, te in enumerate(depth_df["test_acc"]):
        ax.annotate(f"{te*100:.1f}%", (xi, te * 100), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(depth_df.index.tolist())
    ax.set_xlabel("max_depth", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Decision Tree: max_depth vs Accuracy", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(50, 105)
    plt.tight_layout()
    save("depth_experiment")


# ── Plot 6: Model Comparison Bar Chart ───────────────────────────────────────
def plot_model_comparison():
    path = f"{MODELS_DIR}/model_accuracies.csv"
    if not os.path.exists(path):
        print("model_accuracies.csv not found, skipping")
        return
    acc_df = pd.read_csv(path)
    colors = ["#9b59b6", "#e74c3c", "#2ecc71", "#3498db"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(acc_df["model"], acc_df["accuracy"] * 100,
                  color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, acc_df["accuracy"] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=12)
    plt.tight_layout()
    save("model_comparison")


# ── Plot 7: MLP Loss Curve (Backpropagation) ──────────────────────────────────
def plot_mlp_loss():
    path = f"{MODELS_DIR}/mlp_loss_curve.csv"
    if not os.path.exists(path):
        print("mlp_loss_curve.csv not found, skipping")
        return
    loss_df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_df["loss"], color="#3498db", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("MLP Neural Network — Training Loss (Backpropagation)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # annotate start/end
    ax.annotate(f"Start: {loss_df['loss'].iloc[0]:.3f}", xy=(0, loss_df['loss'].iloc[0]),
                xytext=(10, 10), textcoords="offset points", fontsize=9, color="#e74c3c")
    ax.annotate(f"End: {loss_df['loss'].iloc[-1]:.4f}", xy=(len(loss_df)-1, loss_df['loss'].iloc[-1]),
                xytext=(-60, 10), textcoords="offset points", fontsize=9, color="#2ecc71")
    plt.tight_layout()
    save("mlp_loss_curve")


# ── Plot 8: MLP Weight Heatmap (Layer 1: 3×64) ───────────────────────────────
def plot_mlp_weights(mlp_bundle):
    mlp = mlp_bundle["model"]
    W1 = mlp.coefs_[0]   # shape (3, 64)

    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(W1.T, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(W1).max(), vmax=np.abs(W1).max())
    ax.set_yticks(range(W1.shape[1]))
    ax.set_yticklabels([f"N{i}" for i in range(W1.shape[1])], fontsize=5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["player_total", "dealer_showing", "usable_ace"], fontsize=11)
    ax.set_xlabel("Input Feature", fontsize=11)
    ax.set_ylabel("Hidden Neuron (64)", fontsize=11)
    ax.set_title("MLP Layer 1 Weights (3 inputs → 64 hidden neurons)\n"
                 "Red = positive weight, Blue = negative weight", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Weight value")
    plt.tight_layout()
    save("mlp_weights")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data and models...")
    df = pd.read_csv(DATASET_PATH)
    dt, mlp_bundle = load_models()

    print("Generating plots...")
    plot_win_rates(df)
    plot_decision_tree(dt)
    plot_feature_importance(dt)
    plot_heatmap(dt, df)
    plot_depth_experiment()
    plot_model_comparison()
    plot_mlp_loss()
    plot_mlp_weights(mlp_bundle)

    print(f"\nAll plots saved to ./{PLOTS_DIR}/")
