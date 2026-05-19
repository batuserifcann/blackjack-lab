"""
train.py — Blackjack Multi-Model Training

Models trained:
  1. Decision Tree        (primary, max_depth=5)
  2. MLP Neural Network   (3 → 64 → 32 → 1)

Training data: analytically generated from basic_strategy() over all valid
(player_total, dealer_showing, usable_ace) states × 500 copies = 140,000 samples.
"""
import os
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
FEATURE_COLS = ["player_total", "dealer_showing", "usable_ace"]


# ── Basic Strategy ────────────────────────────────────────────────────────────

def basic_strategy(player_total, dealer_showing, usable_ace):
    player, dealer, ace = player_total, dealer_showing, usable_ace
    if ace:
        if player <= 17:   return "hit"
        elif player == 18: return "hit" if dealer >= 9 else "stand"
        else:              return "stand"
    if player <= 11:       return "hit"
    elif player == 12:     return "stand" if 4 <= dealer <= 6 else "hit"
    elif 13 <= player <= 16:
                           return "stand" if dealer <= 6 else "hit"
    return "stand"


# ── Dataset ───────────────────────────────────────────────────────────────────

def generate_strategy_dataset(n_copies=500):
    rows = []
    for player_total in range(4, 22):
        for dealer_showing in range(2, 12):
            for usable_ace in [0, 1]:
                if usable_ace and player_total < 12:
                    continue
                action = basic_strategy(player_total, dealer_showing, bool(usable_ace))
                rows.append({
                    "player_total": player_total,
                    "dealer_showing": dealer_showing,
                    "usable_ace": usable_ace,
                    "label": 1 if action == "hit" else 0,
                })
    base_df = pd.DataFrame(rows)
    df = pd.concat([base_df] * n_copies, ignore_index=True)
    print(f"States: {len(base_df)}  |  Samples (×{n_copies}): {len(df):,}")
    print(f"Hit: {df['label'].sum():,}  Stand: {(df['label']==0).sum():,}")
    return df


# ── Model trainers ────────────────────────────────────────────────────────────

def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    dt.fit(X_train, y_train)
    return dt


def train_mlp(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        verbose=False,
    )
    mlp.fit(X_scaled, y_train)
    return mlp, scaler


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(name, model, X_test, y_test, scaler=None):
    X = scaler.transform(X_test) if scaler else X_test
    preds = model.predict(X)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["stand", "hit"]))
    cm = confusion_matrix(y_test, preds)
    print(f"Confusion Matrix:\n{cm}")
    return acc


def depth_experiment(X_train, X_test, y_train, y_test):
    print("\n--- max_depth Experiment (Decision Tree) ---")
    print(f"{'depth':<8} {'train_acc':<12} {'test_acc'}")
    print("-" * 32)
    results = {}
    for depth in [3, 4, 5, 6, 7, None]:
        clf = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        tr = accuracy_score(y_train, clf.predict(X_train))
        te = accuracy_score(y_test, clf.predict(X_test))
        label = str(depth) if depth is not None else "None"
        print(f"{label:<8} {tr:.4f}       {te:.4f}")
        results[label] = {"train_acc": tr, "test_acc": te}
    return results


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(obj, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Generating dataset...")
    df = generate_strategy_dataset(n_copies=500)

    X = df[FEATURE_COLS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Depth experiment
    depth_results = depth_experiment(X_train, X_test, y_train, y_test)
    depth_df = pd.DataFrame(depth_results).T
    depth_df.index.name = "depth"
    depth_df.to_csv(os.path.join(MODELS_DIR, "depth_experiment.csv"))

    print("\n" + "=" * 55)
    print("Training all models...")

    # 1. Decision Tree
    dt = train_decision_tree(X_train, y_train)
    dt_acc = evaluate("Decision Tree (depth=5)", dt, X_test, y_test)
    print("\n  Feature Importances:")
    for feat, imp in zip(FEATURE_COLS, dt.feature_importances_):
        print(f"    {feat:<20} {imp:.4f}")

    # 2. MLP
    mlp, mlp_scaler = train_mlp(X_train, y_train)
    mlp_acc = evaluate("MLP Neural Network (64-32)", mlp, X_test, y_test, mlp_scaler)
    print(f"\n  MLP Architecture:")
    print(f"    Input layer:   3 neurons  (player_total, dealer_showing, usable_ace)")
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        layer_name = f"Hidden {i+1}" if i < len(mlp.coefs_) - 1 else "Output"
        print(f"    {layer_name} layer: {w.shape[0]}→{w.shape[1]} neurons  "
              f"(weights matrix {w.shape[0]}×{w.shape[1]}, bias {b.shape[0]})")
    print(f"    Activation: ReLU  |  Solver: Adam  |  Epochs: {mlp.n_iter_}")
    print(f"    Final loss: {mlp.loss_:.6f}")

    # Summary
    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print(f"{'Model':<28} {'Accuracy'}")
    print("-" * 40)
    for name, acc in [
        ("Decision Tree (depth=5)", dt_acc),
        ("MLP Neural Network (64-32)", mlp_acc),
    ]:
        bar = "█" * int(acc * 30)
        print(f"  {name:<26} {acc:.4f}  {bar}")

    # Save everything
    save_model(dt, "decision_tree")
    save_model({"model": mlp, "scaler": mlp_scaler}, "mlp")

    # Save MLP loss curve for visualize.py
    loss_df = pd.DataFrame({"loss": mlp.loss_curve_})
    loss_df.to_csv(os.path.join(MODELS_DIR, "mlp_loss_curve.csv"), index=False)

    # Save model accuracies for visualize.py
    acc_df = pd.DataFrame({
        "model": ["Decision Tree", "MLP Neural Net"],
        "accuracy": [dt_acc, mlp_acc],
    })
    acc_df.to_csv(os.path.join(MODELS_DIR, "model_accuracies.csv"), index=False)

    print(f"\nAll models saved to ./{MODELS_DIR}/")
