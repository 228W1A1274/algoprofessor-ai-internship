import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DealHunter Leaderboard", page_icon="🏆", layout="wide")
st.title("🏆 DealHunter — LLM Price Oracle Leaderboard")
st.markdown("**Milestone 7 | AlgoProfessor AI R&D Internship 2026 | Phase 2**")
st.markdown("---")

lb = Path("output/leaderboard.json")
if not lb.exists():
    st.warning("No results found. Run evaluator.py first.")
    st.stop()

with open(lb) as f:
    data = json.load(f)

valid = [r for r in data if r.get("mape") is not None]
df = pd.DataFrame(valid).sort_values("mape").reset_index(drop=True)
medals = ["🥇", "🥈", "🥉"]
df.insert(0, "Rank", [medals[i] if i < len(medals) else "" for i in range(len(df))])

st.subheader("📊 Model Leaderboard")
st.dataframe(
    df[["Rank", "model", "mae", "mape", "rmse", "accuracy_within_20pct", "parse_success_rate"]],
    use_container_width=True,
    hide_index=True,
)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(df, x="model", y="mape", color="model", title="MAPE (lower=better)")
    fig.add_hline(y=20, line_dash="dash", annotation_text="Target 20%")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = px.bar(df, x="model", y="mae", color="model", title="MAE $ (lower=better)")
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("DealHunter | AlgoProfessor AI R&D 2026 | Milestone 7")
