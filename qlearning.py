"""
qlearning.py — Blackjack Q-Learning Agent

Trains purely from game rewards — no labeled data.
State: (player_total, dealer_showing, usable_ace)
Actions: 0=stand, 1=hit
Q-table: dict mapping state → [Q_stand, Q_hit]

Update rule:
  Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') − Q(s,a)]

Run: .venv/bin/python qlearning.py
"""
import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game_engine import Deck, BlackjackGame, hand_value, card_value

MODELS_DIR = "models"
PLOTS_DIR  = "plots"

# ── Hyperparameters ───────────────────────────────────────────────────────────
ALPHA   = 0.1    # learning rate
GAMMA   = 0.95   # discount factor (single-step game → near 1 is fine)
EPSILON_START = 1.0   # exploration rate start
EPSILON_END   = 0.05  # minimum exploration
EPSILON_DECAY = 0.999995  # per-episode decay
N_EPISODES    = 500_000
EVAL_EVERY    = 5_000   # evaluate win rate every N episodes
EVAL_GAMES    = 500     # games per evaluation

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Q-table ───────────────────────────────────────────────────────────────────

def state_key(player_total, dealer_showing, usable_ace):
    return (int(player_total), int(dealer_showing), int(usable_ace))


def get_q(q_table, state):
    if state not in q_table:
        q_table[state] = [0.0, 0.0]   # [Q_stand, Q_hit]
    return q_table[state]


def choose_action(q_table, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)    # explore
    q = get_q(q_table, state)
    return int(np.argmax(q))          # exploit


# ── Single episode ────────────────────────────────────────────────────────────

def run_episode(deck, q_table, epsilon, alpha, gamma):
    game = BlackjackGame(deck)
    game.deal_initial()

    if game.done:
        # immediate blackjack / dealer blackjack
        reward = 1.5 if game.result == "blackjack" else (0 if game.result == "draw" else -1)
        return reward

    transitions = []   # (state, action) pairs this episode

    while not game.done:
        s = game.get_state()
        key = state_key(s["player_total"], s["dealer_showing"], s["usable_ace"])
        action = choose_action(q_table, key, epsilon)
        transitions.append((key, action))

        if action == 1:    # hit
            _, bust = game.player_hit()
            if bust:
                break
        else:              # stand
            game.player_stand()
            break

    result = game.result
    reward = 1.5 if result == "blackjack" else (1 if result == "win" else (0 if result == "draw" else -1))

    # Backward Q-update (single-step game: only terminal reward matters)
    for key, action in transitions:
        q = get_q(q_table, key)
        q[action] += alpha * (reward - q[action])   # γ=1 for terminal

    return reward


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(q_table, n_games=500):
    deck = Deck(num_decks=6)
    wins = 0
    for _ in range(n_games):
        game = BlackjackGame(deck)
        game.deal_initial()
        if not game.done:
            while not game.done:
                s = game.get_state()
                key = state_key(s["player_total"], s["dealer_showing"], s["usable_ace"])
                action = choose_action(q_table, key, epsilon=0.0)
                if action == 1:
                    _, bust = game.player_hit()
                    if bust:
                        break
                else:
                    game.player_stand()
                    break
        if game.result in ("win", "blackjack"):
            wins += 1
    return wins / n_games * 100


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    print(f"Q-Learning Training")
    print(f"  Episodes : {N_EPISODES:,}")
    print(f"  α={ALPHA}  γ={GAMMA}  ε {EPSILON_START}→{EPSILON_END}")
    print(f"  Eval every {EVAL_EVERY:,} episodes ({EVAL_GAMES} games)\n")

    q_table = {}
    epsilon = EPSILON_START
    deck = Deck(num_decks=6)

    eval_points = []   # (episode, win_rate)
    epsilon_log = []

    for ep in range(1, N_EPISODES + 1):
        run_episode(deck, q_table, epsilon, ALPHA, GAMMA)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if ep % EVAL_EVERY == 0:
            wr = evaluate(q_table, EVAL_GAMES)
            eval_points.append((ep, wr))
            epsilon_log.append(epsilon)
            print(f"  ep {ep:>7,}  ε={epsilon:.4f}  win_rate={wr:.1f}%  "
                  f"states={len(q_table)}")

    print(f"\nTraining done. Q-table states: {len(q_table)}")
    return q_table, eval_points, epsilon_log


# ── Policy extraction ─────────────────────────────────────────────────────────

def extract_policy(q_table):
    """Show learned policy vs basic strategy for key states."""
    from simulate import basic_strategy as bs

    print("\n--- Learned Policy vs Basic Strategy ---")
    print(f"{'State':<35} {'Q-Learn':<10} {'Basic':<10} {'Match'}")
    print("-" * 60)
    matches = total = 0
    for player in range(8, 22):
        for dealer in range(2, 12):
            for ace in [0, 1]:
                if ace and player < 12:
                    continue
                key = state_key(player, dealer, bool(ace))
                q = get_q(q_table, key)
                ql_action = "hit" if np.argmax(q) == 1 else "stand"
                bs_action = bs({"player_total": player, "dealer_showing": dealer, "usable_ace": bool(ace)})
                match = "✔" if ql_action == bs_action else "✘"
                if ql_action == bs_action:
                    matches += 1
                total += 1
    print(f"\nPolicy match with basic strategy: {matches}/{total} = {matches/total*100:.1f}%")
    return matches / total * 100


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_convergence(eval_points):
    episodes = [e for e, _ in eval_points]
    win_rates = [w for _, w in eval_points]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(episodes, win_rates, color="#3498db", linewidth=2, label="Q-Learning win rate")
    ax.axhline(43, color="#2ecc71", linestyle="--", linewidth=1.5, alpha=0.8, label="Basic strategy ~43%")
    ax.axhline(29, color="#e67e22", linestyle=":", linewidth=1.2, alpha=0.6, label="Random ~29%")

    # shade exploration → exploitation transition
    mid = episodes[len(episodes) // 3]
    ax.axvspan(episodes[0], mid, alpha=0.08, color="#e74c3c", label="Exploration phase")
    ax.axvspan(mid, episodes[-1], alpha=0.08, color="#2ecc71", label="Exploitation phase")

    ax.set_xlabel("Episode", color="#aaa", fontsize=11)
    ax.set_ylabel("Win Rate (%)", color="#aaa", fontsize=11)
    ax.set_title("Q-Learning Convergence", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="#ccc", framealpha=0.9)
    ax.set_ylim(0, 60)
    fig.tight_layout()
    path = f"{PLOTS_DIR}/ql_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_policy_heatmap(q_table):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0e1117")

    from simulate import basic_strategy as bs
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list("bj", ["#2ecc71", "#e74c3c"])

    for ax_i, (title, use_ql) in enumerate([
        ("Basic Strategy", False),
        ("Q-Learning Policy", True),
    ]):
        ax = axes[ax_i]
        ax.set_facecolor("#0e1117")
        player_totals  = range(8, 22)
        dealer_showings = range(2, 12)
        grid = np.zeros((len(player_totals), len(dealer_showings)))

        for i, pt in enumerate(player_totals):
            for j, ds in enumerate(dealer_showings):
                if use_ql:
                    key = state_key(pt, ds, 0)
                    q = get_q(q_table, key)
                    grid[i, j] = float(np.argmax(q))   # 1=hit, 0=stand
                else:
                    action = bs({"player_total": pt, "dealer_showing": ds, "usable_ace": False})
                    grid[i, j] = 1 if action == "hit" else 0

        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(dealer_showings)))
        ax.set_xticklabels([str(d) if d < 11 else "A" for d in dealer_showings], color="#ccc")
        ax.set_yticks(range(len(player_totals)))
        ax.set_yticklabels(list(player_totals), color="#ccc")
        ax.set_xlabel("Dealer Showing", color="#aaa", fontsize=10)
        ax.set_ylabel("Player Total", color="#aaa", fontsize=10)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")

    fig.suptitle("Hit (red) vs Stand (green) — Basic Strategy vs Q-Learning",
                 color="white", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=axes, label="Action (1=Hit, 0=Stand)")
    fig.tight_layout()
    path = f"{PLOTS_DIR}/ql_policy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    q_table, eval_points, epsilon_log = train()

    policy_match = extract_policy(q_table)

    # Final win rate
    final_wr = evaluate(q_table, n_games=5000)
    print(f"\nFinal win rate (5,000 games, greedy): {final_wr:.1f}%")

    # Save
    with open(f"{MODELS_DIR}/q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print(f"Saved: {MODELS_DIR}/q_table.pkl")

    pd.DataFrame(eval_points, columns=["episode", "win_rate"]).to_csv(
        f"{MODELS_DIR}/ql_convergence.csv", index=False
    )

    # Plots
    plot_convergence(eval_points)
    plot_policy_heatmap(q_table)

    print(f"\nSummary:")
    print(f"  Q-table states   : {len(q_table)}")
    print(f"  Policy vs basic  : {policy_match:.1f}%")
    print(f"  Final win rate   : {final_wr:.1f}%")
    print(f"  Basic strategy   : ~43%")
