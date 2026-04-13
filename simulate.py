import csv
import os
import random
from game_engine import Deck, BlackjackGame, card_value


# ---- Stratejiler ----

def random_strategy(state):
    """Tamamen rastgele oyna."""
    return random.choice(["hit", "stand"])


def naive_strategy(state):
    """Basit kural: 17'nin altinda hit, ustunde stand."""
    if state["player_total"] < 17:
        return "hit"
    return "stand"


def basic_strategy(state):
    """Kitaplardaki Basic Strategy tablosunun basitlestirilmis hali."""
    player = state["player_total"]
    dealer = state["dealer_showing"]
    ace = state["usable_ace"]

    if ace:
        if player <= 17:
            return "hit"
        elif player == 18:
            if dealer >= 9:
                return "hit"
            return "stand"
        else:
            return "stand"

    if player <= 11:
        return "hit"
    elif player == 12:
        if 4 <= dealer <= 6:
            return "stand"
        return "hit"
    elif 13 <= player <= 16:
        if dealer <= 6:
            return "stand"
        return "hit"
    else:
        return "stand"


# ---- Simulasyon ----

def simulate(num_games, strategy_fn, strategy_name):
    deck = Deck(num_decks=6)
    results = []

    for i in range(num_games):
        game = BlackjackGame(deck)
        result = game.play_one_hand(strategy_fn)
        result["episode_id"] = i + 1
        result["strategy"] = strategy_name
        results.append(result)

        if (i + 1) % 25000 == 0:
            print(f"  {strategy_name}: {i+1:,} / {num_games:,} tamamlandi")

    return results


def save_to_csv(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = [
        "episode_id", "strategy", "player_total", "dealer_total",
        "dealer_showing", "usable_ace", "num_hits", "actions",
        "result", "reward"
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nKaydedildi: {filepath} ({len(results):,} satir)")


# ---- Calistir ----

if __name__ == "__main__":
    all_results = []

    print("Simulasyon basliyor...\n")

    r1 = simulate(100000, random_strategy, "random")
    all_results.extend(r1)

    r2 = simulate(100000, naive_strategy, "naive")
    all_results.extend(r2)

    r3 = simulate(100000, basic_strategy, "basic")
    all_results.extend(r3)

    save_to_csv(all_results, "dataset/games.csv")

    print("\n--- Sonuclar ---")
    for name, data in [("Random", r1), ("Naive", r2), ("Basic", r3)]:
        wins = sum(1 for r in data if r["result"] in ("win", "blackjack"))
        print(f"  {name:<10} Kazanma: {wins/len(data)*100:.1f}%")