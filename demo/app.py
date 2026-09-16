from pathlib import Path

import streamlit as st

from demo.session_data import best_candidate, list_examples, load_example

EXAMPLES_DIR = Path(__file__).parent / "examples"

st.set_page_config(page_title="art-director-agent", page_icon="🎨", layout="wide")
st.title("art-director-agent")
st.caption(
    "Replays real critique-loop sessions: generation, defect diagnosis, fix, "
    "and (when accepted) upscale. No live generation happens here — this is a "
    "viewer over sessions the agent already ran."
)

slugs = list_examples(EXAMPLES_DIR)
if not slugs:
    st.warning("No example sessions found in demo/examples/.")
    st.stop()

slug = st.selectbox("Example session", slugs)
session = load_example(EXAMPLES_DIR, slug)
winner = best_candidate(session)

st.subheader("Brief")
st.write(session.brief)

for round_num, round_candidates in enumerate(session.rounds, start=1):
    st.subheader(f"Round {round_num}")
    cols = st.columns(len(round_candidates))
    for col, candidate in zip(cols, round_candidates, strict=True):
        with col:
            st.image(str(EXAMPLES_DIR / candidate["path"]), width="stretch")
            score = candidate.get("clip_score")
            score_label = f"CLIP {score:.1f}" if score is not None else "CLIP: n/a"
            is_best = winner is not None and candidate["path"] == winner["path"]
            st.caption(f"{score_label}{' ⭐ best in session' if is_best else ''}")
            if "source_path" in candidate:
                st.caption(f"fix of round-{round_num - 1} candidate, region {candidate['bbox']}")

st.divider()

if session.upscaled_path:
    st.subheader("Accepted — upscaled final output")
    st.image(str(EXAMPLES_DIR / session.upscaled_path), width="stretch")
else:
    st.subheader("Not upscaled")
    st.info(
        "This session's best candidate was never upscaled in this curated set — "
        "either the loop hit its round cap without a confirmed pass (ADR 0005), "
        "or the defect (e.g. text rendering) was a known architectural limitation "
        "no fix in this taxonomy addresses. Both are honest, documented outcomes, "
        "not failures of the demo."
    )
