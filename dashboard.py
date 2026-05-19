"""
Blackjack AI Dashboard — Streamlit
Run: .venv/bin/streamlit run dashboard.py
"""
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

from game_engine import Deck, BlackjackGame, hand_value, card_value

st.set_page_config(page_title="Blackjack AI", page_icon="🃏", layout="wide")

# ── Load all models ───────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    import os
    models = {}
    paths = {
        "Decision Tree":      ("models/decision_tree.pkl",      False),
        "MLP Neural Network": ("models/mlp.pkl",                 True),
    }
    for name, (path, is_bundle) in paths.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if is_bundle:
                models[name] = {"model": obj["model"], "scaler": obj["scaler"]}
            else:
                models[name] = {"model": obj, "scaler": None}

    # Q-Learning
    ql_path = "models/q_table.pkl"
    if os.path.exists(ql_path):
        with open(ql_path, "rb") as f:
            models["Q-Learning (RL)"] = {"model": pickle.load(f), "scaler": None}

    return models

ALL_MODELS = load_all_models()

MODEL_INFO = {
    "Decision Tree": {
        "accuracy": "99.3%",
        "type": "Tree (primary)",
        "color": "#e74c3c",
        "note": "Interpretable if-else rules. Optimal depth=5 balances accuracy and compactness.",
    },
    "MLP Neural Network": {
        "accuracy": "100%",
        "type": "Neural Net (3→64→32→1)",
        "color": "#3498db",
        "note": "Learns via backpropagation. ReLU activations. Weights updated with Adam optimizer.",
    },
    "Q-Learning (RL)": {
        "accuracy": "86.2% policy match",
        "type": "Reinforcement Learning",
        "color": "#f39c12",
        "note": "No labeled data. Learns from win/lose rewards via Q(s,a) ← Q(s,a) + α[r − Q(s,a)].",
    },
}


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
        "selected_model": "Decision Tree",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── AI prediction ─────────────────────────────────────────────────────────────
def ai_recommend(player_total, dealer_showing, usable_ace):
    model_name = st.session_state.selected_model
    bundle = ALL_MODELS.get(model_name)
    if bundle is None:
        return "hit", 0.5, 0.5

    model  = bundle["model"]
    scaler = bundle["scaler"]

    # Q-Learning
    if model_name == "Q-Learning (RL)":
        key = (int(player_total), int(dealer_showing), int(usable_ace))
        q = model.get(key, [0.0, 0.0])
        action = "hit" if np.argmax(q) == 1 else "stand"
        total = abs(q[0]) + abs(q[1]) + 1e-9
        hit_prob = (q[1] + abs(min(q[1], 0))) / (total + abs(min(q[1], 0)) + abs(min(q[0], 0)) + 1e-9)
        hit_prob = float(np.clip(hit_prob, 0.01, 0.99))
        return action, hit_prob, 1 - hit_prob

    feat = pd.DataFrame(
        [[player_total, dealer_showing, int(usable_ace)]],
        columns=["player_total", "dealer_showing", "usable_ace"],
    )
    X = scaler.transform(feat) if scaler else feat
    pred = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        hit_prob, stand_prob = float(prob[1]), float(prob[0])
    else:
        hit_prob  = 0.85 if pred == 1 else 0.15
        stand_prob = 1 - hit_prob
    action = "hit" if pred == 1 else "stand"
    return action, hit_prob, stand_prob


def explain_decision(player_total, dealer_showing, usable_ace, action):
    p, d, ace = player_total, dealer_showing, bool(usable_ace)
    if d in (2, 3):   dealer_desc, dealer_strong = f"dealer shows **{d}** (weak)", False
    elif d in (4,5,6):dealer_desc, dealer_strong = f"dealer shows **{d}** (very weak — ~42% bust)", False
    elif d in (7, 8): dealer_desc, dealer_strong = f"dealer shows **{d}** (moderate)", True
    else:
        label = "Ace" if d == 11 else str(d)
        dealer_desc, dealer_strong = f"dealer shows **{label}** (strong)", True

    lines = [f"📊 **State:** Player=`{p}` {'*(soft)*' if ace else ''} | {dealer_desc}"]

    if ace:
        lines.append("🂡 **Soft hand** — Ace counted as 11. Cannot bust on next card.")
        if p <= 17:   lines.append(f"📌 **Rule:** Soft {p} → **HIT** (no bust risk, improve hand).")
        elif p == 18: lines.append(f"📌 **Rule:** Soft 18 vs {'strong' if dealer_strong else 'weak'} dealer → **{action.upper()}**.")
        else:          lines.append(f"📌 **Rule:** Soft {p} → **STAND** (strong hand).")
    elif p <= 11:  lines.append(f"📌 **Rule:** {p} ≤ 11 → **HIT** (impossible to bust).")
    elif p == 12:
        if 4 <= d <= 6: lines.append("📌 **Rule:** 12 vs weak dealer (4–6) → **STAND** (let dealer bust).")
        else:            lines.append("📌 **Rule:** 12 vs strong dealer → **HIT** (12 too weak to stand).")
    elif 13 <= p <= 16:
        if not dealer_strong: lines.append(f"📌 **Rule:** {p} vs weak dealer → **STAND** (dealer bust ~35–42%).")
        else:                  lines.append(f"📌 **Rule:** {p} vs strong dealer → **HIT** (standing likely loses).")
    else:
        lines.append(f"📌 **Rule:** {p} ≥ 17 → **STAND** (hitting risks bust).")

    return "\n\n".join(lines)


# ── Card rendering ────────────────────────────────────────────────────────────
SUIT_COLOR = {"♠": "#1a1a2e", "♣": "#1a1a2e", "♥": "#c0392b", "♦": "#c0392b"}
SUIT_BG    = {"♠": "#eaf0fb", "♣": "#eaf0fb", "♥": "#fdecea", "♦": "#fdecea"}

def card_html(rank, suit, hidden=False):
    if hidden:
        return ('<div style="display:inline-block;width:62px;height:90px;border-radius:10px;'
                'background:repeating-linear-gradient(45deg,#2c3e50,#2c3e50 6px,#34495e 6px,#34495e 12px);'
                'border:2px solid #1a252f;margin:4px;vertical-align:middle;"></div>')
    c, bg = SUIT_COLOR.get(suit,"#000"), SUIT_BG.get(suit,"#fff")
    return (f'<div style="display:inline-block;width:62px;height:90px;border-radius:10px;'
            f'background:{bg};border:2px solid #bdc3c7;margin:4px;vertical-align:middle;'
            f'text-align:center;box-shadow:2px 3px 8px rgba(0,0,0,0.18);position:relative;">'
            f'<div style="position:absolute;top:5px;left:7px;font-size:15px;font-weight:700;color:{c};">{rank}</div>'
            f'<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:26px;color:{c};">{suit}</div>'
            f'<div style="position:absolute;bottom:5px;right:7px;font-size:15px;font-weight:700;color:{c};transform:rotate(180deg);">{rank}</div>'
            f'</div>')

def hand_html(hand, hide_second=False):
    return "".join(card_html("","",hidden=True) if (hide_second and i==1) else card_html(r,s)
                   for i,(r,s) in enumerate(hand))


# ── Stats helpers ─────────────────────────────────────────────────────────────
def win_rate():
    h = st.session_state.history
    return sum(1 for r in h if r in ("win","blackjack")) / len(h) * 100 if h else 0.0

def follow_rate():
    log = st.session_state.decision_log
    return sum(1 for d in log if d["followed"]) / len(log) * 100 if log else 0.0

def win_rate_chart():
    history = st.session_state.history
    if len(history) < 2:
        return None
    wins, cumulative = 0, []
    for i, r in enumerate(history):
        if r in ("win","blackjack"): wins += 1
        cumulative.append(wins / (i+1) * 100)
    fig, ax = plt.subplots(figsize=(5, 2.4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    color = MODEL_INFO.get(st.session_state.selected_model, {}).get("color", "#3498db")
    ax.plot(cumulative, color=color, linewidth=1.8, label="You")
    ax.axhline(43, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.7, label="Basic ~43%")
    ax.axhline(29, color="#e67e22", linestyle=":", linewidth=1, alpha=0.5, label="Random ~29%")
    ax.set_ylim(0, 80)
    ax.set_xlabel("Hand #", color="#aaa", fontsize=8)
    ax.set_ylabel("Win %", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa", labelsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="#ccc", framealpha=0.8)
    fig.tight_layout(pad=0.4)
    return fig


# ── Game actions ──────────────────────────────────────────────────────────────
def new_hand():
    st.session_state.hand_num += 1
    st.session_state.last_explanation = None
    game = BlackjackGame(st.session_state.deck)
    game.deal_initial()
    st.session_state.game = game
    if game.done: finish_hand()
    else:         st.session_state.phase = "playing"

def _record(state, ai_action, player_action):
    expl = explain_decision(state["player_total"], state["dealer_showing"], state["usable_ace"], ai_action)
    st.session_state.last_explanation = expl
    st.session_state.decision_log.append({
        "hand": st.session_state.hand_num,
        "player_total": state["player_total"],
        "dealer_showing": state["dealer_showing"],
        "ai": ai_action, "player_action": player_action,
        "followed": ai_action == player_action,
    })

def do_hit():
    game = st.session_state.game
    state = game.get_state()
    ai_action, _, _ = ai_recommend(state["player_total"], state["dealer_showing"], state["usable_ace"])
    _record(state, ai_action, "hit")
    game.player_hit()
    if game.done: finish_hand()

def do_stand():
    game = st.session_state.game
    state = game.get_state()
    ai_action, _, _ = ai_recommend(state["player_total"], state["dealer_showing"], state["usable_ace"])
    _record(state, ai_action, "stand")
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
.stButton > button { width:100%;height:52px;font-size:18px;font-weight:700;border-radius:10px; }
.result-box { padding:14px 20px;border-radius:12px;font-size:22px;font-weight:800;text-align:center;margin:10px 0; }
.ai-box     { padding:12px 18px;border-radius:10px;font-size:20px;font-weight:700;text-align:center;margin:8px 0; }
.explain-box{ background:#1a1f2e;border-left:4px solid #3498db;border-radius:8px;padding:14px 18px;margin-top:10px;font-size:14px;line-height:1.7; }
.model-badge{ display:inline-block;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:700;margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — Model Selector ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Model")
    available = list(ALL_MODELS.keys())
    selected = st.radio("Select model:", available,
                        index=available.index(st.session_state.selected_model)
                              if st.session_state.selected_model in available else 0)

    if selected != st.session_state.selected_model:
        st.session_state.selected_model = selected
        # reset stats on model change
        st.session_state.history = []
        st.session_state.decision_log = []
        st.session_state.hand_num = 0
        st.session_state.game = None
        st.session_state.phase = "idle"

    info = MODEL_INFO.get(selected, {})
    color = info.get("color", "#888")
    st.markdown(f'<span class="model-badge" style="background:{color};color:white;">{info.get("type","")}</span>',
                unsafe_allow_html=True)
    st.markdown(f"**Accuracy:** {info.get('accuracy','?')}")
    st.caption(info.get("note",""))

    st.divider()
    st.markdown("### 📖 Model Comparison")
    comp_data = {
        "Model": ["DT", "MLP", "QL"],
        "Accuracy": ["99.3%", "100%", "86.2%*"],
    }
    st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)
    st.caption("*Policy match vs basic strategy")

    st.divider()
    with st.expander("📖 House Edge"):
        st.markdown("""
Even with perfect AI, player loses ~48% of hands.

Player acts **before** dealer — busting loses immediately regardless of dealer's cards. This gives the casino ~0.5% structural edge.

Theoretical maximum (with card counting): **~50.5%**
        """)

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown(f"## 🃏 Blackjack AI — **{st.session_state.selected_model}**")
st.caption(f"{info.get('type','')} | Accuracy: {info.get('accuracy','?')} | Win rate ceiling: ~43%")

col_game, col_explain, col_stats = st.columns([2, 2, 2], gap="large")

game  = st.session_state.game
phase = st.session_state.phase

# ── GAME column ───────────────────────────────────────────────────────────────
with col_game:
    st.markdown("#### Dealer")
    if game is None:
        st.markdown(card_html("","",hidden=True)*2, unsafe_allow_html=True)
    elif phase == "playing":
        st.markdown(hand_html(game.dealer_hand, hide_second=True), unsafe_allow_html=True)
        st.caption(f"Showing: **{card_value(game.dealer_hand[0])}**  |  1 card hidden")
    else:
        d_total, _ = hand_value(game.dealer_hand)
        st.markdown(hand_html(game.dealer_hand), unsafe_allow_html=True)
        st.caption(f"Total: **{d_total}**")

    st.divider()

    st.markdown("#### You")
    if game is None:
        st.markdown(card_html("","",hidden=True)*2, unsafe_allow_html=True)
    else:
        p_total, p_ace = hand_value(game.player_hand)
        st.markdown(hand_html(game.player_hand), unsafe_allow_html=True)
        st.caption(f"Total: **{p_total}**{' *(soft)*' if p_ace else ''}")

    st.divider()

    if phase == "playing":
        state = game.get_state()
        ai_action, hit_prob, stand_prob = ai_recommend(
            state["player_total"], state["dealer_showing"], state["usable_ace"]
        )
        bg  = "#d4edda" if ai_action == "hit" else "#fff3cd"
        fg  = "#155724" if ai_action == "hit" else "#856404"
        emoji = "⬆️ HIT" if ai_action == "hit" else "■ STAND"
        st.markdown(f'<div class="ai-box" style="background:{bg};color:{fg};">'
                    f'{emoji}<br><span style="font-size:13px;font-weight:400;">'
                    f'Hit: {hit_prob*100:.0f}%  |  Stand: {stand_prob*100:.0f}%'
                    f'</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("🔴 Hit", f"{hit_prob*100:.0f}%"); c1.progress(hit_prob)
        c2.metric("🟢 Stand", f"{stand_prob*100:.0f}%"); c2.progress(stand_prob)

    if phase == "done" and game is not None:
        p_total, _ = hand_value(game.player_hand)
        d_total, _ = hand_value(game.dealer_hand)
        result = game.result
        if result in ("win","blackjack"): bg,fg,label = "#d4edda","#155724","✔ WIN"+(" — BLACKJACK!" if result=="blackjack" else "")
        elif result == "lose":            bg,fg,label = "#f8d7da","#721c24","✘ LOSE"
        else:                             bg,fg,label = "#fff3cd","#856404","― DRAW"
        st.markdown(f'<div class="result-box" style="background:{bg};color:{fg};">'
                    f'{label}<br><span style="font-size:14px;font-weight:400;">You {p_total} — Dealer {d_total}</span>'
                    f'</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🂠 Deal",  disabled=(phase=="playing"),  use_container_width=True): new_hand();  st.rerun()
    with b2:
        if st.button("⬆️ Hit",  disabled=(phase!="playing"),  use_container_width=True): do_hit();   st.rerun()
    with b3:
        if st.button("■ Stand", disabled=(phase!="playing"),  use_container_width=True): do_stand(); st.rerun()

# ── EXPLANATION column ────────────────────────────────────────────────────────
with col_explain:
    st.markdown("#### 🧠 Why did AI say that?")

    if phase == "playing" and game is not None:
        state = game.get_state()
        ai_action, hit_prob, stand_prob = ai_recommend(
            state["player_total"], state["dealer_showing"], state["usable_ace"]
        )
        expl = explain_decision(state["player_total"], state["dealer_showing"], state["usable_ace"], ai_action)
        st.markdown(f'<div class="explain-box">{expl.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

        st.markdown("**Feature input:**")
        st.dataframe(pd.DataFrame({
            "Feature":    ["player_total","dealer_showing","usable_ace"],
            "Value":      [state["player_total"], state["dealer_showing"], "Yes" if state["usable_ace"] else "No"],
            "Importance": ["83%","9%","8%"],
        }), hide_index=True, use_container_width=True)

        st.markdown("**Model call:**")
        st.code(
            f"model = {st.session_state.selected_model}\n"
            f"player_total   = {state['player_total']}\n"
            f"dealer_showing = {state['dealer_showing']}\n"
            f"usable_ace     = {state['usable_ace']}\n\n"
            f"→ predict() = '{ai_action.upper()}'\n"
            f"   P(hit)   = {hit_prob:.2f}\n"
            f"   P(stand) = {stand_prob:.2f}",
            language="python"
        )

    elif st.session_state.last_explanation:
        st.markdown("*Last decision:*")
        st.markdown(f'<div class="explain-box">'
                    f'{st.session_state.last_explanation.replace(chr(10),"<br>")}'
                    f'</div>', unsafe_allow_html=True)
    else:
        st.info("Deal a hand to see AI reasoning here.")

# ── STATS column ──────────────────────────────────────────────────────────────
with col_stats:
    history = st.session_state.history
    total = len(history)
    wins  = sum(1 for r in history if r in ("win","blackjack"))
    draws = history.count("draw")
    losses= history.count("lose")

    st.markdown("#### 📊 Session Stats")
    m1, m2 = st.columns(2)
    m1.metric("Hands", total)
    m2.metric("Win rate", f"{win_rate():.1f}%",
              delta=f"{win_rate()-43:.1f}% vs basic" if total >= 5 else None)

    c1, c2, c3 = st.columns(3)
    c1.metric("✔ Wins",   wins)
    c2.metric("― Draws",  draws)
    c3.metric("✘ Losses", losses)
    st.metric("AI follow rate", f"{follow_rate():.0f}%")

    st.markdown("#### Win Rate Over Time")
    chart = win_rate_chart()
    if chart:
        st.pyplot(chart, use_container_width=True)
    else:
        st.caption("Play a few hands to see the chart.")

    st.markdown("#### Last 10 Decisions")
    log = st.session_state.decision_log[-10:][::-1]
    if log:
        st.dataframe(pd.DataFrame([{
            "Hand": d["hand"], "Player": d["player_total"],
            "Dealer": d["dealer_showing"], "AI": d["ai"].upper(),
            "You": d["player_action"].upper(), "✔": "✔" if d["followed"] else "✘",
        } for d in log]), hide_index=True, use_container_width=True, height=240)
    else:
        st.caption("No decisions yet.")
