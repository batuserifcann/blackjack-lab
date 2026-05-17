"""
Interactive Blackjack with AI Advisor
--------------------------------------
Play real hands. AI shows recommended action at every turn.
Press Enter to follow AI suggestion, or type 'h'/'s' to override.
Type 'q' to quit.
"""
import os
import pickle
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

from game_engine import Deck, BlackjackGame, hand_value, card_value

MODEL_PATH = "models/decision_tree.pkl"

# ── ANSI colors ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

SUITS_COLOR = {"♠": "", "♣": "", "♥": RED, "♦": RED}


def colored_card(rank, suit):
    c = SUITS_COLOR.get(suit, "")
    return f"{c}{rank}{suit}{RESET}"


def format_hand(hand):
    return "  ".join(colored_card(r, s) for r, s in hand)


def ai_recommend(model, player_total, dealer_showing, usable_ace):
    feat = pd.DataFrame(
        [[player_total, dealer_showing, int(usable_ace)]],
        columns=["player_total", "dealer_showing", "usable_ace"],
    )
    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0]
    action = "hit" if pred == 1 else "stand"
    confidence = prob[pred]
    return action, confidence


def print_banner():
    print(f"\n{BOLD}{'─'*52}")
    print(f"  ♠ ♥  BLACKJACK  +  AI ADVISOR  ♦ ♣")
    print(f"{'─'*52}{RESET}")


def print_state(game, hide_dealer=True):
    d_hand = game.dealer_hand
    p_hand = game.player_hand
    p_total, p_ace = hand_value(p_hand)
    d_visible = card_value(d_hand[0])

    print(f"\n{DIM}{'─'*52}{RESET}")
    if hide_dealer:
        d_str = colored_card(*d_hand[0]) + f"  {DIM}[hidden]{RESET}"
        print(f"  Dealer:  {d_str}    {DIM}(showing {d_visible}){RESET}")
    else:
        d_total, _ = hand_value(d_hand)
        print(f"  Dealer:  {format_hand(d_hand)}    total={d_total}")

    ace_tag = f" {DIM}(soft){RESET}" if p_ace else ""
    print(f"  You:     {format_hand(p_hand)}    {BOLD}total={p_total}{RESET}{ace_tag}")


def print_ai(action, confidence):
    color = GREEN if action == "hit" else YELLOW
    arrow = "HIT  ↑" if action == "hit" else "STAND ■"
    print(f"\n  {CYAN}AI says:{RESET}  {color}{BOLD}{arrow}{RESET}  {DIM}({confidence*100:.0f}% confident){RESET}")


def print_result(result, player_total, dealer_total):
    if result in ("win", "blackjack"):
        tag = f"{GREEN}{BOLD}  ✔  YOU WIN{' — BLACKJACK!' if result == 'blackjack' else ''}{RESET}"
    elif result == "lose":
        tag = f"{RED}{BOLD}  ✘  YOU LOSE{RESET}"
    else:
        tag = f"{YELLOW}{BOLD}  ─  DRAW{RESET}"
    print(f"\n{tag}   (you={player_total}, dealer={dealer_total})")


def play_session(model):
    deck = Deck(num_decks=6)
    stats = {"win": 0, "lose": 0, "draw": 0, "blackjack": 0, "hands": 0}
    followed_ai = 0
    total_decisions = 0

    print_banner()
    print(f"  {DIM}Enter = follow AI  |  h = hit  |  s = stand  |  q = quit{RESET}\n")

    while True:
        game = BlackjackGame(deck)
        game.deal_initial()
        stats["hands"] += 1

        print(f"\n{BOLD}Hand #{stats['hands']}{RESET}")
        print_state(game, hide_dealer=True)

        # ── Immediate blackjack / dealer blackjack ───────────────────────
        if game.done:
            p_total, _ = hand_value(game.player_hand)
            d_total, _ = hand_value(game.dealer_hand)
            print(f"\n  {CYAN}Dealer reveals:{RESET}  {format_hand(game.dealer_hand)}  total={d_total}")
            print_result(game.result, p_total, d_total)
            stats[game.result] += 1
            _print_stats(stats, followed_ai, total_decisions)
            inp = input(f"\n  {DIM}Next hand? [Enter / q]{RESET}  ").strip().lower()
            if inp == "q":
                break
            continue

        # ── Player turn ──────────────────────────────────────────────────
        while not game.done:
            state = game.get_state()
            rec, conf = ai_recommend(
                model, state["player_total"], state["dealer_showing"], state["usable_ace"]
            )
            print_ai(rec, conf)

            raw = input(f"  {DIM}[Enter={rec} / h / s / q]{RESET}  ").strip().lower()

            if raw == "q":
                _print_final(stats, followed_ai, total_decisions)
                return

            if raw == "":
                action = rec
                followed_ai += 1
            elif raw == "h":
                action = "hit"
            elif raw == "s":
                action = "stand"
            else:
                print(f"  {DIM}Unknown input — treated as follow AI ({rec}){RESET}")
                action = rec
                followed_ai += 1

            total_decisions += 1

            if action == "hit":
                card, bust = game.player_hit()
                print(f"  Drew: {colored_card(*card)}", end="")
                if bust:
                    p_total, _ = hand_value(game.player_hand)
                    print(f"  → {RED}{BOLD}BUST ({p_total}){RESET}")
                else:
                    p_total, p_ace = hand_value(game.player_hand)
                    ace_tag = f" {DIM}(soft){RESET}" if p_ace else ""
                    print(f"  → total={BOLD}{p_total}{RESET}{ace_tag}")
            else:
                game.player_stand()

        # ── Show result ──────────────────────────────────────────────────
        p_total, _ = hand_value(game.player_hand)
        d_total, _ = hand_value(game.dealer_hand)
        print(f"\n  {CYAN}Dealer reveals:{RESET}  {format_hand(game.dealer_hand)}  total={d_total}")
        print_result(game.result, p_total, d_total)
        stats[game.result] += 1
        _print_stats(stats, followed_ai, total_decisions)

        inp = input(f"\n  {DIM}Next hand? [Enter / q]{RESET}  ").strip().lower()
        if inp == "q":
            break

    _print_final(stats, followed_ai, total_decisions)


def _print_stats(stats, followed, total):
    h = stats["hands"]
    wins = stats["win"] + stats["blackjack"]
    win_rate = wins / h * 100 if h else 0
    follow_rate = followed / total * 100 if total else 0
    print(
        f"  {DIM}[ Hands: {h}  |  W:{stats['win']+stats['blackjack']}  "
        f"D:{stats['draw']}  L:{stats['lose']}  |  "
        f"Win rate: {win_rate:.1f}%  |  AI followed: {follow_rate:.0f}% ]{RESET}"
    )


def _print_final(stats, followed, total):
    h = stats["hands"]
    wins = stats["win"] + stats["blackjack"]
    print(f"\n{BOLD}{'─'*52}")
    print(f"  SESSION OVER")
    print(f"{'─'*52}{RESET}")
    print(f"  Hands played : {h}")
    print(f"  Wins         : {wins}  ({wins/h*100:.1f}%)" if h else "  Wins: 0")
    print(f"  Draws        : {stats['draw']}")
    print(f"  Losses       : {stats['lose']}")
    if total:
        print(f"  AI followed  : {followed}/{total} ({followed/total*100:.0f}%)")
    print()


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run train.py first.")
        raise SystemExit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    play_session(model)
