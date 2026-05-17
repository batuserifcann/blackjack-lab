"""
Blackjack AI Dashboard — Streamlit
Run: .venv/bin/streamlit run dashboard.py
"""
import pickle
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

from game_engine import Deck, BlackjackGame, hand_value, card_value

MODEL_PATH = "models/decision_tree.pkl"

st.set_page_config(page_title="Blackjack AI", page_icon="🃏", layout="wide")

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "deck": Deck(num_decks=6),
        "game": None,
        "phase": "idle",
        "history": [],
        "decision_log": [],
        "hand_num": 0,
        "last_explanation": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── AI ────────────────────────────────────────────────────────────────────────
def ai_recommend(player_total, dealer_showing, usable_ace):
    feat = pd.DataFrame(
        [[player_total, dealer_showing, int(usable_ace)]],
        columns=["player_total", "dealer_showing", "usable_ace"],
    )
    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0]
    action = "hit" if pred == 1 else "stand"
    return action, float(prob[1]), float(prob[0])   # action, hit_prob, stand_prob


def explain_decision(player_total, dealer_showing, usable_ace, action):
    """Return plain-language explanation of the AI's decision."""
    p, d, ace = player_total, dealer_showing, bool(usable_ace)

    # Dealer strength bucket
    if d in (2, 3):
        dealer_desc = f"dealer shows **{d}** (weak — likely to bust)"
        dealer_strong = False
    elif d in (4, 5, 6):
        dealer_desc = f"dealer shows **{d}** (very weak — high bust probability)"
        dealer_strong = False
    elif d in (7, 8):
        dealer_desc = f"dealer shows **{d}** (moderate strength)"
        dealer_strong = True
    else:
        label = "Ace" if d == 11 else str(d)
        dealer_desc = f"dealer shows **{label}** (strong — likely to make 17+)"
        dealer_strong = True

    lines = [f"📊 **State:** Player total = `{p}` {'*(soft/ace)*' if ace else ''} | {dealer_desc}"]

    # Rule explanation
    if ace:
        lines.append("🂡 **Soft hand** (Ace counted as 11) — can always hit safely, no instant bust risk.")
        if p <= 17:
            lines.append(f"📌 **Rule:** Soft {p} → always **HIT** (can't bust, improves hand).")
        elif p == 18:
            if d >= 9:
                lines.append(f"📌 **Rule:** Soft 18 vs strong dealer ({d}) → **HIT** (18 may not beat dealer's likely 19–20).")
            else:
                lines.append(f"📌 **Rule:** Soft 18 vs weak dealer → **STAND** (18 is strong enough).")
        else:
            lines.append(f"📌 **Rule:** Soft {p} → **STAND** (already a strong hand).")
    elif p <= 11:
        lines.append(f"📌 **Rule:** Total {p} ≤ 11 → always **HIT** (impossible to bust on next card).")
    elif p == 12:
        if 4 <= d <= 6:
            lines.append(f"📌 **Rule:** 12 vs weak dealer (4–6) → **STAND** (let dealer bust themselves).")
        else:
            lines.append(f"📌 **Rule:** 12 vs strong dealer → **HIT** (12 is too weak to risk standing).")
    elif 13 <= p <= 16:
        if not dealer_strong:
            lines.append(f"📌 **Rule:** {p} vs weak dealer (2–6) → **STAND** (dealer has high bust chance ~42%).")
        else:
            lines.append(f"📌 **Rule:** {p} vs strong dealer → **HIT** (standing on {p} almost certainly loses).")
    else:
        lines.append(f"📌 **Rule:** {p} ≥ 17 → always **STAND** (hitting risks busting on a pat hand).")

    # House edge note
    lines.append("")
    if action == "stand" and not dealer_strong:
        lines.append("🏠 *Dealer must hit until ≥17 — weak upcard means ~35–42% bust probability. Patience wins here.*")
    elif action == "hit" and dealer_strong:
        lines.append("🏠 *Dealer upcard suggests a strong hidden hand. Taking a card is statistically better than surrendering.*")

    return "\n\n".join(lines)


# ── Card rendering ────────────────────────────────────────────────────────────
SUIT_COLOR = {"♠": "#1a1a2e", "♣": "#1a1a2e", "♥": "#c0392b", "♦": "#c0392b"}
SUIT_BG    = {"♠": "#eaf0fb", "♣": "#eaf0fb", "♥": "#fdecea", "♦": "#fdecea"}

def render_card_html(rank, suit, hidden=False):
    if hidden:
        return (
            '<div style="display:inline-block;width:62px;height:90px;border-radius:10px;'
            'background:repeating-linear-gradient(45deg,#2c3e50,#2c3e50 6px,#34495e 6px,#34495e 12px);'
            'border:2px solid #1a252f;margin:4px;vertical-align:middle;"></div>'
        )
    color = SUIT_COLOR.get(suit, "#000")
    bg    = SUIT_BG.get(suit, "#fff")
    return (
        f'<div style="display:inline-block;width:62px;height:90px;border-radius:10px;'
        f'background:{bg};border:2px solid #bdc3c7;margin:4px;vertical-align:middle;'
        f'text-align:center;box-shadow:2px 3px 8px rgba(0,0,0,0.18);position:relative;">'
        f'<div style="position:absolute;top:5px;left:7px;font-size:15px;font-weight:700;color:{color};">{rank}</div>'
        f'<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:26px;color:{color};">{suit}</div>'
        f'<div style="position:absolute;bottom:5px;right:7px;font-size:15px;font-weight:700;color:{color};transform:rotate(180deg);">{rank}</div>'
        f'</div>'
    )

def render_hand_html(hand, hide_second=False):
    return "".join(
        render_card_html("", "", hidden=True) if (hide_second and i == 1)
        else render_card_html(r, s)
        for i, (r, s) in enumerate(hand)
    )


# ── Stats ─────────────────────────────────────────────────────────────────────
def win_rate():
    h = st.session_state.history
    if not h:
        return 0.0
    return sum(1 for r in h if r in ("win", "blackjack")) / len(h) * 100

def follow_rate():
    log = st.session_state.decision_log
    if not log:
        return 0.0
    return sum(1 for d in log if d["followed"]) / len(log) * 100

def win_rate_chart():
    history = st.session_state.history
    if len(history) < 2:
        return None
    cumulative, wins = [], 0
    for i, r in enumerate(history):
        if r in ("win", "blackjack"):
            wins += 1
        cumulative.append(wins / (i + 1) * 100)

    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot(cumulative, color="#2ecc71", linewidth=1.8, label="You")
    ax.axhline(43, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="Basic strategy ~43%")
    ax.axhline(29, color="#e67e22", linestyle=":", linewidth=1, alpha=0.5, label="Random ~29%")
    ax.set_ylim(0, 80)
    ax.set_xlabel("Hand #", color="#aaa", fontsize=8)
    ax.set_ylabel("Win %", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#ccc", framealpha=0.8)
    fig.tight_layout(pad=0.4)
    return fig

def decision_table():
    log = st.session_state.decision_log[-10:][::-1]
    if not log:
        return None
    rows = [{
        "Hand": d["hand"],
        "Player": d["player_total"],
        "Dealer": d["dealer_showing"],
        "AI": d["ai"].upper(),
        "You": d["player_action"].upper(),
        "✔": "✔" if d["followed"] else "✘",
    } for d in log]
    return pd.DataFrame(rows)


# ── Game actions ──────────────────────────────────────────────────────────────
def new_hand():
    st.session_state.hand_num += 1
    st.session_state.last_explanation = None
    game = BlackjackGame(st.session_state.deck)
    game.deal_initial()
    st.session_state.game = game
    if game.done:
        finish_hand()
    else:
        st.session_state.phase = "playing"

def _record_decision(state, ai_action, player_action):
    expl = explain_decision(
        state["player_total"], state["dealer_showing"], state["usable_ace"], ai_action
    )
    st.session_state.last_explanation = expl
    st.session_state.decision_log.append({
        "hand": st.session_state.hand_num,
        "player_total": state["player_total"],
        "dealer_showing": state["dealer_showing"],
        "ai": ai_action,
        "player_action": player_action,
        "followed": ai_action == player_action,
    })

def do_hit():
    game = st.session_state.game
    state = game.get_state()
    ai_action, _, _ = ai_recommend(state["player_total"], state["dealer_showing"], state["usable_ace"])
    _record_decision(state, ai_action, "hit")
    game.player_hit()
    if game.done:
        finish_hand()

def do_stand():
    game = st.session_state.game
    state = game.get_state()
    ai_action, _, _ = ai_recommend(state["player_total"], state["dealer_showing"], state["usable_ace"])
    _record_decision(state, ai_action, "stand")
    game.player_stand()
    finish_hand()

def finish_hand():
    st.session_state.history.append(st.session_state.game.result)
    st.session_state.phase = "done"


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stButton > button {
    width:100%; height:52px; font-size:18px; font-weight:700;
    border-radius:10px; border:none; cursor:pointer;
}
.result-box {
    padding:14px 20px; border-radius:12px; font-size:22px;
    font-weight:800; text-align:center; margin:10px 0;
}
.ai-box {
    padding:12px 18px; border-radius:10px; font-size:20px;
    font-weight:700; text-align:center; margin:8px 0;
}
.explain-box {
    background:#1a1f2e; border-left:4px solid #3498db;
    border-radius:8px; padding:14px 18px; margin-top:10px;
    font-size:14px; line-height:1.7;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🃏 Blackjack AI — Decision Explainer")
st.caption("Decision Tree (99.3% accuracy) | ~43% win rate = mathematical ceiling with basic strategy")

col_game, col_explain, col_stats = st.columns([2, 2, 2], gap="large")

game  = st.session_state.game
phase = st.session_state.phase

# ── GAME COLUMN ───────────────────────────────────────────────────────────────
with col_game:
    st.markdown("#### Dealer")
    if game is None:
        st.markdown(render_card_html("", "", hidden=True) * 2, unsafe_allow_html=True)
    elif phase == "playing":
        st.markdown(render_hand_html(game.dealer_hand, hide_second=True), unsafe_allow_html=True)
        st.caption(f"Showing: **{card_value(game.dealer_hand[0])}**  |  1 card hidden")
    else:
        d_total, _ = hand_value(game.dealer_hand)
        st.markdown(render_hand_html(game.dealer_hand), unsafe_allow_html=True)
        st.caption(f"Total: **{d_total}**")

    st.divider()

    st.markdown("#### You")
    if game is None:
        st.markdown(render_card_html("", "", hidden=True) * 2, unsafe_allow_html=True)
    else:
        p_total, p_ace = hand_value(game.player_hand)
        st.markdown(render_hand_html(game.player_hand), unsafe_allow_html=True)
        ace_tag = " *(soft)*" if p_ace else ""
        st.caption(f"Total: **{p_total}**{ace_tag}")

    st.divider()

    # AI recommendation box
    if phase == "playing":
        state = game.get_state()
        ai_action, hit_prob, stand_prob = ai_recommend(
            state["player_total"], state["dealer_showing"], state["usable_ace"]
        )
        if ai_action == "hit":
            bg, fg, emoji = "#d4edda", "#155724", "⬆️ HIT"
        else:
            bg, fg, emoji = "#fff3cd", "#856404", "■ STAND"

        st.markdown(
            f'<div class="ai-box" style="background:{bg};color:{fg};">'
            f'AI recommends: {emoji}'
            f'<br><span style="font-size:13px;font-weight:400;">'
            f'Hit prob: {hit_prob*100:.0f}%  |  Stand prob: {stand_prob*100:.0f}%'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        # probability bar
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("🔴 Hit", f"{hit_prob*100:.0f}%")
            st.progress(hit_prob)
        with prob_col2:
            st.metric("🟢 Stand", f"{stand_prob*100:.0f}%")
            st.progress(stand_prob)

    # Result
    if phase == "done" and game is not None:
        p_total, _ = hand_value(game.player_hand)
        d_total, _ = hand_value(game.dealer_hand)
        result = game.result
        if result in ("win", "blackjack"):
            bg, fg, label = "#d4edda", "#155724", "✔ WIN" + (" — BLACKJACK!" if result == "blackjack" else "")
        elif result == "lose":
            bg, fg, label = "#f8d7da", "#721c24", "✘ LOSE"
        else:
            bg, fg, label = "#fff3cd", "#856404", "― DRAW"
        st.markdown(
            f'<div class="result-box" style="background:{bg};color:{fg};">'
            f'{label}<br>'
            f'<span style="font-size:14px;font-weight:400;">You {p_total} — Dealer {d_total}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Buttons
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🂠 Deal", disabled=(phase == "playing"), use_container_width=True):
            new_hand(); st.rerun()
    with b2:
        if st.button("⬆️ Hit", disabled=(phase != "playing"), use_container_width=True):
            do_hit(); st.rerun()
    with b3:
        if st.button("■ Stand", disabled=(phase != "playing"), use_container_width=True):
            do_stand(); st.rerun()

# ── EXPLANATION COLUMN ────────────────────────────────────────────────────────
with col_explain:
    st.markdown("#### 🧠 Why did AI say that?")

    if phase == "playing" and game is not None:
        state = game.get_state()
        ai_action, hit_prob, stand_prob = ai_recommend(
            state["player_total"], state["dealer_showing"], state["usable_ace"]
        )
        expl = explain_decision(
            state["player_total"], state["dealer_showing"], state["usable_ace"], ai_action
        )
        st.markdown(
            f'<div class="explain-box">{expl.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

        # Decision tree feature breakdown
        st.markdown("**Feature breakdown:**")
        feat_data = {
            "Feature": ["Player total", "Dealer showing", "Has soft Ace"],
            "Value": [
                state["player_total"],
                state["dealer_showing"],
                "Yes" if state["usable_ace"] else "No",
            ],
            "Importance": ["83%", "9%", "8%"],
        }
        st.dataframe(pd.DataFrame(feat_data), hide_index=True, use_container_width=True)

        st.markdown("**Decision Tree logic (depth 1→5):**")
        st.code(
            f"player_total = {state['player_total']}\n"
            f"dealer_showing = {state['dealer_showing']}\n"
            f"usable_ace = {state['usable_ace']}\n"
            f"\n→ model.predict() = '{ai_action.upper()}'\n"
            f"   hit_probability  = {hit_prob:.2f}\n"
            f"   stand_probability = {stand_prob:.2f}",
            language="python",
        )

    elif st.session_state.last_explanation:
        st.markdown("*Last decision:*")
        st.markdown(
            f'<div class="explain-box">'
            f'{st.session_state.last_explanation.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    else:
        st.info("Deal a hand to see the AI's reasoning here.")

    # Static house edge explainer
    with st.expander("📖 Why does the dealer win more often?"):
        st.markdown("""
**Blackjack house edge explained:**

Even with *perfect* basic strategy, the mathematical win rates are:
| Outcome | Probability |
|---------|------------|
| Player wins | ~43% |
| Dealer wins | ~49% |
| Draw | ~8% |

**Why?** The player acts first. If the player busts, the dealer wins automatically — even if the dealer would have busted too. This structural rule gives the house a ~0.5% edge regardless of strategy.

**Can we beat 43%?** Only with card counting: tracking which cards remain in the shoe shifts probabilities slightly. Theoretical maximum with perfect card counting ≈ **50.5%** — just barely over 50%.

The AI is performing optimally. The losses are mathematically inevitable, not a mistake.
        """)

# ── STATS COLUMN ─────────────────────────────────────────────────────────────
with col_stats:
    st.markdown("#### 📊 Session Stats")

    history = st.session_state.history
    total   = len(history)
    wins    = sum(1 for r in history if r in ("win","blackjack"))
    draws   = history.count("draw")
    losses  = history.count("lose")

    m1, m2 = st.columns(2)
    m1.metric("Hands", total)
    m2.metric("Win rate", f"{win_rate():.1f}%", delta=f"{win_rate()-43:.1f}% vs basic" if total else None)

    c1, c2, c3 = st.columns(3)
    c1.metric("✔ Wins", wins)
    c2.metric("― Draws", draws)
    c3.metric("✘ Losses", losses)

    st.metric("AI follow rate", f"{follow_rate():.0f}%")

    st.markdown("#### Win Rate Over Time")
    chart = win_rate_chart()
    if chart:
        st.pyplot(chart, use_container_width=True)
    else:
        st.caption("Play a few hands to see the chart.")

    st.markdown("#### Last 10 Decisions")
    df_log = decision_table()
    if df_log is not None:
        st.dataframe(df_log, hide_index=True, use_container_width=True, height=240)
    else:
        st.caption("No decisions yet.")
