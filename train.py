"""
train.py — Blackjack Decision Tree / Random Forest Training

Training data is generated directly from the basic_strategy function over
all possible (player_total, dealer_showing, usable_ace) states. This is
the correct approach: the dataset's player_total column records the FINAL
hand value after all hits, not the value at decision time, so using it as
a training feature would create a systematic mismatch.
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODELS_DIR = "models"


# ---- Basic Strategy (copied from simulate.py to avoid cross-dependency) ----

def basic_strategy(player_total, dealer_showing, usable_ace):
    player = player_total
    dealer = dealer_showing
    ace = usable_ace
    if ace:
        if player <= 17:
            return "hit"
        elif player == 18:
            return "hit" if dealer >= 9 else "stand"
        return "stand"
    if player <= 11:
        return "hit"
    elif player == 12:
        return "stand" if 4 <= dealer <= 6 else "hit"
    elif 13 <= player <= 16:
        return "stand" if dealer <= 6 else "hit"
    return "stand"


# ---- Dataset Generation ----

def generate_strategy_dataset(n_copies=500):
    """
    Generate labeled training data from all valid game states.
    Replicated n_copies times to give classifiers enough samples.
    """
    rows = []
    for player_total in range(4, 22):
        for dealer_showing in range(2, 12):
            for usable_ace in [0, 1]:
                # usable ace only meaningful when player has an ace counted as 11
                if usable_ace and player_total < 12:
                    continue  # can't have soft hand below 12
                action = basic_strategy(player_total, dealer_showing, bool(usable_ace))
                label = 1 if action == "hit" else 0
                rows.append({
                    "player_total": player_total,
                    "dealer_showing": dealer_showing,
                    "usable_ace": usable_ace,
                    "label": label,
                })
    base_df = pd.DataFrame(rows)
    df = pd.concat([base_df] * n_copies, ignore_index=True)
    print(f"States: {len(base_df)}  |  Training samples (×{n_copies}): {len(df)}")
    print(f"Hit: {df['label'].sum():,}  Stand: {(df['label']==0).sum():,}")
    return df


# ---- Depth Experiment ----

def depth_experiment(X_train, X_test, y_train, y_test):
    print("\n--- max_depth Experiment ---")
    print(f"{'depth':<10} {'train_acc':<12} {'test_acc':<12}")
    print("-" * 34)
    results = {}
    for depth in [3, 4, 5, 6, 7, None]:
        clf = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        label = str(depth) if depth is not None else "None"
        print(f"{label:<10} {train_acc:.4f}       {test_acc:.4f}")
        results[label] = {"train_acc": train_acc, "test_acc": test_acc}
    return results


# ---- Models ----

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    return clf


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["stand", "hit"]))
    cm = confusion_matrix(y_test, preds)
    print(f"Confusion Matrix:\n{cm}")
    return acc


def save_model(model, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating strategy dataset from basic_strategy function...\n")
    df = generate_strategy_dataset(n_copies=500)

    feature_cols = ["player_total", "dealer_showing", "usable_ace"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    depth_results = depth_experiment(X_train, X_test, y_train, y_test)

    print("\n--- Training Models ---")
    dt = train_decision_tree(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    evaluate("Decision Tree (max_depth=5)", dt, X_test, y_test)
    evaluate("Random Forest (n=100)", rf, X_test, y_test)

    print("\n--- Feature Importances (Random Forest) ---")
    for feat, imp in zip(feature_cols, rf.feature_importances_):
        print(f"  {feat:<20} {imp:.4f}")

    save_model(dt, "decision_tree")
    save_model(rf, "random_forest")

    depth_df = pd.DataFrame(depth_results).T
    depth_df.index.name = "depth"
    depth_df.to_csv(os.path.join(MODELS_DIR, "depth_experiment.csv"))
    print(f"\nDone. Models in ./{MODELS_DIR}/")
