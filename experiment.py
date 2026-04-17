"""
experiment.py  —  Stroop Color-Word A/B Test (Between-Subject)
================================================================
Runs the interactive Stroop experiment in the browser via Streamlit.

USAGE:
    streamlit run experiment.py

HOW IT WORKS:
    - Each participant is randomly and silently assigned to Group A or Group B.
    - Group A (Control)   : sees only CONGRUENT trials   (word meaning == font color)
    - Group B (Treatment) : sees only INCONGRUENT trials (word meaning != font color)
    - Participants complete 40 trials and download a CSV at the end.
    - The researcher collects all CSVs and runs analysis.py.

REQUIREMENTS:
    pip install streamlit scipy matplotlib numpy requests
"""


import streamlit as st
import time
import random
import csv
import io
import threading
import requests
from datetime import datetime

# ── Google Sheets auto-submission URL ──────────────────────────────────────────
SHEETS_URL = (
    "https://script.google.com/macros/s/"
    "AKfycby9AjFcx5ajC4PNwwx80DRs8Ws88kZP7B4JAE35yb0wIjabierO1KSiLLixeHc27jJUCg/exec"
)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Color Identification Task",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Constants ──────────────────────────────────────────────────────────────────
COLORS: dict[str, str] = {
    "RED":    "#E53935",
    "BLUE":   "#1E88E5",
    "GREEN":  "#43A047",
    "YELLOW": "#F9A825",
    "PURPLE": "#8E24AA",
    "PINK":   "#E91E63",
}
COLOR_NAMES = list(COLORS.keys())
NUM_TRIALS  = 40   # each participant does 40 trials of ONE condition only

# ── CSS: colored response buttons via aria-label ───────────────────────────────
st.markdown("""
<style>
  .main .block-container { max-width: 660px; padding-top: 1.8rem; }
  footer { visibility: hidden; }

  button[aria-label="RED"]    { background:#E53935!important; color:white!important;  font-weight:bold!important; border:none!important; }
  button[aria-label="BLUE"]   { background:#1E88E5!important; color:white!important;  font-weight:bold!important; border:none!important; }
  button[aria-label="GREEN"]  { background:#43A047!important; color:white!important;  font-weight:bold!important; border:none!important; }
  button[aria-label="YELLOW"] { background:#F9A825!important; color:#333!important;   font-weight:bold!important; border:none!important; }
  button[aria-label="PURPLE"] { background:#8E24AA!important; color:white!important;  font-weight:bold!important; border:none!important; }
  button[aria-label="PINK"]   { background:#E91E63!important; color:white!important;  font-weight:bold!important; border:none!important; }

  .stroop-word {
    font-size: 96px; font-weight: 900; text-align: center;
    padding: 44px 0 10px; line-height: 1;
    font-family: Helvetica, Arial, sans-serif;
  }
  .trial-hint {
    text-align: center; color: #BDBDBD;
    font-style: italic; font-size: 14px; margin-bottom: 22px;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────
def _init_state():
    """Initialise all session-state keys with default values."""
    defaults = dict(
        stage         = "welcome",    # welcome → trial → results
        group         = None,         # "A" or "B" — assigned once, never shown to participant
        condition     = None,         # "congruent" or "incongruent"
        user_id       = "",
        trials        = [],
        current_trial = 0,
        results       = [],
        trial_start   = 0.0,
        last_rendered = -1,
        btn_order     = [],
        submitted     = False,        # prevent duplicate Google Sheets submissions
    )
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()
ss = st.session_state


# ── Helper functions ───────────────────────────────────────────────────────────
def _assign_group() -> tuple[str, str]:
    """
    Randomly assign participant to Group A (congruent) or Group B (incongruent).
    Assignment is 50/50 and not revealed to the participant.
    """
    group     = random.choice(["A", "B"])
    condition = "congruent" if group == "A" else "incongruent"
    return group, condition


def _generate_trials(condition: str) -> list[dict]:
    """
    Generate NUM_TRIALS trials for a single condition.

    Congruent  (Group A): word meaning == display color  e.g. "RED" shown in red
    Incongruent(Group B): word meaning != display color  e.g. "RED" shown in blue
    """
    trials = []
    for _ in range(NUM_TRIALS):
        if condition == "congruent":
            color = random.choice(COLOR_NAMES)
            trials.append({
                "word":      color,
                "color":     color,
                "condition": "congruent",
                "answer":    color,
            })
        else:
            word  = random.choice(COLOR_NAMES)
            color = random.choice([c for c in COLOR_NAMES if c != word])
            trials.append({
                "word":      word,
                "color":     color,
                "condition": "incongruent",
                "answer":    color,
            })
    random.shuffle(trials)
    return trials


def _record_response(response: str):
    """
    Record the participant's response for the current trial.
    Called via Streamlit button on_click callback.
    """
    rt_ms   = (time.perf_counter() - ss.trial_start) * 1000
    trial   = ss.trials[ss.current_trial]
    correct = response == trial["answer"]

    ss.results.append({
        "user_id":          ss.user_id,
        "group":            ss.group,       # A or B
        "trial_num":        ss.current_trial + 1,
        "condition":        trial["condition"],
        "word":             trial["word"],
        "display_color":    trial["color"],
        "response":         response,
        "correct":          correct,
        "reaction_time_ms": round(rt_ms, 2),
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    ss.current_trial += 1
    ss.last_rendered  = -1   # reset timer on next render


def _to_csv(results: list[dict]) -> bytes:
    """Serialise results list to CSV bytes for download."""
    buf = io.StringIO()
    if results:
        writer = csv.DictWriter(buf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    return buf.getvalue().encode()


def _submit_to_sheets(results: list[dict]):
    """
    Silently POST all trial rows to Google Sheets in a background thread.
    If the request fails for any reason, the CSV download still works as backup.
    """
    def _send():
        try:
            requests.post(SHEETS_URL, json=results, timeout=15)
        except Exception:
            pass   # fail silently — participant can still download CSV

    threading.Thread(target=_send, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Welcome screen
# ══════════════════════════════════════════════════════════════════════════════
if ss.stage == "welcome":
    st.markdown("## 🎨 Color Identification Task")
    st.caption("Reaction Time Study")

    with st.container(border=True):
        st.markdown("""
**Instructions**

A color word will appear on screen in a colored font.
**Click the button matching the FONT COLOR.**
Ignore what the word says — respond to the ink color only.

📊 &nbsp;40 trials &nbsp;·&nbsp; ~3 minutes &nbsp;·&nbsp; respond as fast and accurately as you can
        """)

    uid = st.text_input(
        "Participant ID",
        value=f"P{random.randint(100, 999)}",
        max_chars=20,
    )

    if st.button("Start →", type="primary", use_container_width=True):
        # Assign group (hidden from participant)
        group, condition      = _assign_group()
        ss.group              = group
        ss.condition          = condition
        ss.user_id            = uid.strip() or f"P{random.randint(100, 999)}"
        ss.trials             = _generate_trials(condition)
        ss.current_trial      = 0
        ss.results            = []
        ss.last_rendered      = -1
        # Fix button order once for the entire experiment (never shuffled again)
        fixed_order = COLOR_NAMES[:]
        random.shuffle(fixed_order)
        ss.btn_order          = fixed_order
        ss.stage              = "trial"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Trial screen
# ══════════════════════════════════════════════════════════════════════════════
elif ss.stage == "trial":
    idx    = ss.current_trial
    trials = ss.trials

    # Auto-advance to results when all trials are done
    if idx >= len(trials):
        ss.stage = "results"
        st.rerun()

    trial = trials[idx]

    # Set timer once per trial (not on every Streamlit rerun caused by button click)
    if ss.last_rendered != idx:
        ss.trial_start   = time.perf_counter()
        ss.last_rendered = idx
        # btn_order is fixed for the whole experiment — do NOT shuffle here

    # Progress bar
    st.markdown(
        f'<div style="text-align:right;font-size:12px;color:#9E9E9E;margin-bottom:4px;">'
        f'Trial {idx + 1} / {len(trials)}</div>',
        unsafe_allow_html=True,
    )
    st.progress(idx / len(trials))

    # Stroop word (large, colored)
    st.markdown(
        f'<div class="stroop-word" style="color:{COLORS[trial["color"]]};">'
        f'{trial["word"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="trial-hint">Click the color of the FONT — ignore the word</div>',
        unsafe_allow_html=True,
    )

    # Response buttons — 2 rows × 3 columns, order shuffled each trial
    order = ss.btn_order
    for row_start in [0, 3]:
        cols = st.columns(3)
        for i, col in enumerate(cols):
            name = order[row_start + i]
            col.button(
                name,
                key=f"btn_{name}_{idx}",
                use_container_width=True,
                on_click=_record_response,
                args=(name,),
            )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Results screen
# ══════════════════════════════════════════════════════════════════════════════
elif ss.stage == "results":
    results = ss.results

    # Auto-submit to Google Sheets once (background thread, silent on failure)
    if not ss.submitted and results:
        _submit_to_sheets(results)
        ss.submitted = True

    # Quick personal summary (does not reveal group condition)
    correct_results = [r for r in results if r["correct"]]
    avg_rt  = round(sum(r["reaction_time_ms"] for r in correct_results) / len(correct_results)) \
              if correct_results else 0
    acc_pct = round(len(correct_results) / len(results) * 100) if results else 0

    st.markdown("## ✅ All done!")
    st.caption(f"Participant: **{ss.user_id}** · {len(results)} trials completed")

    col1, col2 = st.columns(2)
    col1.metric("Avg Response Time", f"{avg_rt} ms")
    col2.metric("Accuracy", f"{acc_pct}%")

    st.info(
        "📄 Please click **Download CSV** below and send the file to your researcher.",
        icon="📋",
    )

    # CSV download
    csv_fname = f"stroop_{ss.group}_{ss.user_id}.csv"
    st.download_button(
        label="⬇ Download CSV",
        data=_to_csv(results),
        file_name=csv_fname,
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )

    # Restart button
    if st.button("Start over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
