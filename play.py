"""
AI Blackjack Simulation
Loads trained Decision Tree model and plays 1000 games.
Compares win rate vs basic strategy benchmark.
"""
import os
import pickle
import warnings
import pandas as pd
from game_engine import Deck, BlackjackGame

warnings.filterwarnings("ignore", category=UserWarning)


BASIC_STRATEGY_WIN_RATE = None  # computed from dataset if available
MODEL_PATH = "models/decision_tree.pkl"
NUM_GAMES = 1000


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}  —  run train.py first")
    with open(path, "rb") as f:
        return pickle.load(f)


def ai_strategy(model):
    def _strategy(state):
        features = pd.DataFrame([[state["player_total"], state["dealer_showing"], int(state["usable_ace"])]],
                                columns=["player_total", "dealer_showing", "usable_ace"])
        pred = model.predict(features)[0]
        return "hit" if pred == 1 else "stand"
    return _strategy


def basic_strategy(state):
    player = state["player_total"]
    dealer = state["dealer_showing"]
    ace = state["usable_ace"]
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


def simulate_games(strategy_fn, num_games, label):
    deck = Deck(num_decks=6)
    wins = draws = losses = 0
    for _ in range(num_games):
        game = BlackjackGame(deck)
        result = game.play_one_hand(strategy_fn)
        if result["result"] in ("win", "blackjack"):
            wins += 1
        elif result["result"] == "draw":
            draws += 1
        else:
            losses += 1
    win_rate = wins / num_games * 100
    draw_rate = draws / num_games * 100
    loss_rate = losses / num_games * 100
    print(f"\n{label} ({num_games:,} games):")
    print(f"  Win:  {wins:>5}  ({win_rate:.1f}%)")
    print(f"  Draw: {draws:>5}  ({draw_rate:.1f}%)")
    print(f"  Loss: {losses:>5}  ({loss_rate:.1f}%)")
    return win_rate


def print_single_game(model):
    """Show one sample game with AI decisions printed."""
    deck = Deck(num_decks=6)
    game = BlackjackGame(deck)
    game.deal_initial()

    from game_engine import hand_value, card_value
    print("\n--- Sample Game ---")
    print(f"Dealer shows: {card_value(game.dealer_hand[0])}")
    print(f"Player hand:  {game.player_hand}  (total={hand_value(game.player_hand)[0]})")

    turn = 0
    while not game.done:
        state = game.get_state()
        features = pd.DataFrame(
            [[state["player_total"], state["dealer_showing"], int(state["usable_ace"])]],
            columns=["player_total", "dealer_showing", "usable_ace"]
        )
        pred = model.predict(features)[0]
        action = "hit" if pred == 1 else "stand"
        prob = model.predict_proba(features)[0]
        print(f"  Turn {turn+1}: player={state['player_total']}  dealer_showing={state['dealer_showing']}"
              f"  ace={state['usable_ace']}  →  AI says '{action}'"
              f"  (hit_prob={prob[1]:.2f})")
        if action == "hit":
            game.player_hit()
        else:
            game.player_stand()
            break
        turn += 1

    from game_engine import hand_value
    p_total, _ = hand_value(game.player_hand)
    d_total, _ = hand_value(game.dealer_hand)
    print(f"  Result: {game.result.upper()}  (player={p_total}, dealer={d_total})")


if __name__ == "__main__":
    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print_single_game(model)

    print(f"\n{'='*40}")
    print("Win Rate Comparison")
    print('='*40)

    ai_rate = simulate_games(ai_strategy(model), NUM_GAMES, "AI (Decision Tree)")
    basic_rate = simulate_games(basic_strategy, NUM_GAMES, "Basic Strategy (benchmark)")

    print(f"\nAI vs Basic Strategy: {ai_rate:.1f}% vs {basic_rate:.1f}%")
    diff = ai_rate - basic_rate
    direction = "better" if diff >= 0 else "worse"
    print(f"AI is {abs(diff):.1f}pp {direction} than basic strategy in this run.")
