"""
pdf_generator.py
================
MeetScribe — PDF Report & Analytics Module
AlgoProfessor AI Internship | Milestone 4 | Day 25

Responsibilities:
    1. Generate meeting analytics charts (speaker time, timeline, LLM comparison)
    2. Export a professional PDF meeting minutes report using ReportLab
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image as RLImage
)
from reportlab.lib import colors


# ─────────────────────────────────────────────────────────
# 1. COLOUR PALETTE
# ─────────────────────────────────────────────────────────

PALETTE = [
    "#1F3A8C",  # Deep blue     (SPEAKER_00 / primary)
    "#E85D04",  # Orange        (SPEAKER_01 / secondary)
    "#2E7D32",  # Green         (SPEAKER_02)
    "#6A1B9A",  # Purple        (SPEAKER_03)
    "#00695C",  # Teal          (SPEAKER_04)
    "#C62828",  # Red           (SPEAKER_05)
    "#F57F17",  # Amber         (SPEAKER_06)
    "#0277BD",  # Light blue    (SPEAKER_07)
]


# ─────────────────────────────────────────────────────────
# 2. ANALYTICS CHARTS
# ─────────────────────────────────────────────────────────

def generate_analytics_charts(
    diarised_transcript: list,
    llm_results: dict,
    output_dir: str = "outputs/"
) -> str:
    """
    Generate a 3-panel analytics dashboard and save as PNG.

    Panels:
        1. Speaker Talk-Time Pie Chart
        2. Speaking Timeline (Gantt-style)
        3. LLM Latency & Cost Comparison Bar Chart

    Args:
        diarised_transcript : Merged speaker+text segments
        llm_results         : Output from run_llm_comparison()
        output_dir          : Directory to save the chart

    Returns:
        Path to saved chart PNG
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, "meeting_analytics.png")

    fig = plt.figure(figsize=(18, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "MeetScribe — Meeting Analytics Dashboard",
        fontsize=14, fontweight="bold", y=1.01, color="#1A1A2E"
    )

    # ── Panel 1: Speaker Talk-Time Pie ────────────────────
    ax1 = fig.add_subplot(1, 3, 1)

    speaker_time = {}
    for seg in diarised_transcript:
        spk = seg["speaker"]
        dur = max(round(seg["end"] - seg["start"], 2), 0.01)
        speaker_time[spk] = speaker_time.get(spk, 0) + dur

    speakers = sorted(speaker_time.keys())
    times    = [speaker_time[s] for s in speakers]
    clrs1    = PALETTE[:len(speakers)]

    wedges, texts, autotexts = ax1.pie(
        times,
        labels=speakers,
        colors=clrs1,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")

    ax1.set_title("Speaker Talk-Time", fontweight="bold", fontsize=11, pad=10)

    # Legend with seconds
    legend_labels = [f"{s}: {t:.0f}s ({t/60:.1f}m)" for s, t in zip(speakers, times)]
    ax1.legend(
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8,
        frameon=False
    )

    # ── Panel 2: Speaking Timeline (Gantt) ────────────────
    ax2 = fig.add_subplot(1, 3, 2)

    color_map = {spk: PALETTE[i % len(PALETTE)] for i, spk in enumerate(speakers)}

    for seg in diarised_transcript:
        width = max(seg["end"] - seg["start"], 0.2)
        ax2.barh(
            seg["speaker"],
            width,
            left=seg["start"],
            color=color_map[seg["speaker"]],
            alpha=0.78,
            height=0.45,
            edgecolor="white",
            linewidth=0.5
        )

    ax2.set_xlabel("Time (seconds)", fontsize=9)
    ax2.set_title("Speaking Timeline", fontweight="bold", fontsize=11, pad=10)
    ax2.grid(axis="x", alpha=0.25, linestyle="--")
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=9)

    # ── Panel 3: LLM Comparison Bar Chart ─────────────────
    ax3 = fig.add_subplot(1, 3, 3)

    successful = {k: v for k, v in llm_results.items()
                  if v.get("status") == "success"}

    if successful:
        model_short = [
            v["model"].replace("(Groq FREE)", "").replace("Sonnet", "").strip()[:16]
            for v in successful.values()
        ]
        latencies = [v["latency_sec"] for v in successful.values()]
        costs     = [v["cost_usd"]    for v in successful.values()]
        clrs3     = PALETTE[:len(model_short)]

        bars = ax3.bar(
            range(len(model_short)), latencies,
            color=clrs3, alpha=0.85,
            edgecolor="white", linewidth=1.5
        )
        ax3.set_xticks(range(len(model_short)))
        ax3.set_xticklabels(model_short, rotation=12, ha="right", fontsize=8)
        ax3.set_ylabel("Latency (seconds)", fontsize=9)
        ax3.set_title("LLM Speed & Cost", fontweight="bold", fontsize=11, pad=10)
        ax3.grid(axis="y", alpha=0.25, linestyle="--")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # Cost annotation on each bar
        for bar, cost in zip(bars, costs):
            label     = "FREE" if cost == 0 else f"${cost:.4f}"
            bar_color = "#2E7D32" if cost == 0 else "#333333"
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(latencies) * 0.02,
                label,
                ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color=bar_color
            )
    else:
        ax3.text(
            0.5, 0.5, "No successful\nLLM results",
            ha="center", va="center",
            transform=ax3.transAxes, fontsize=11, color="grey"
        )
        ax3.set_title("LLM Speed & Cost", fontweight="bold", fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()

    print(f"✅ Analytics chart saved → {chart_path}")
    return chart_path


# ─────────────────────────────────────────────────────────
# 3. PDF REPORT GENERATION
# ─────────────────────────────────────────────────────────

def generate_pdf_report(
    diarised_transcript: list,
    llm_results: dict,
    transcript_info: dict,
    chart_path: str,
    output_path: str = "outputs/meeting_report.pdf",
    meeting_context: str = "Business meeting"
) -> str:
    """
    Generate a professional PDF meeting minutes report.

    Report sections:
        - Header with metadata (date, duration, language, speakers)
        - Analytics dashboard chart
        - Speaker-attributed transcript (first 20 blocks)
        - LLM summaries from all available models
        - Performance comparison table (latency, cost, tokens)

    Args:
        diarised_transcript : Merged speaker+text segments
        llm_results         : Output from run_llm_comparison()
        transcript_info     : Output from transcribe_audio() (for metadata)
        chart_path          : Path to analytics chart PNG
        output_path         : Where to save the PDF
        meeting_context     : Meeting description for the header

    Returns:
        Path to saved PDF file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch
    )

    # ── Define Styles ──────────────────────────────────────
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MTitle", parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1F3A8C"),
        spaceAfter=8,
        spaceBefore=0,
    )
    meta_style = ParagraphStyle(
        "MMeta", parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#666666"),
        spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "MH2", parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#E85D04"),
        spaceAfter=8,
        spaceBefore=14,
        borderPad=3,
    )
    body_style = ParagraphStyle(
        "MBody", parent=styles["Normal"],
        fontSize=9.5,
        spaceAfter=5,
        leading=14,
    )
    speaker_style = ParagraphStyle(
        "MSpk", parent=styles["Normal"],
        fontSize=9.5,
        spaceAfter=4,
        leading=13,
        leftIndent=0,
    )
    code_style = ParagraphStyle(
        "MCode", parent=styles["Code"],
        fontSize=8.5,
        spaceAfter=4,
        leading=12,
        backColor=colors.HexColor("#F5F5F5"),
        borderPad=6,
    )

    story = []

    # ── Header ────────────────────────────────────────────
    now      = datetime.now().strftime("%B %d, %Y  %H:%M")
    dur_min  = transcript_info.get("duration", 0) / 60
    lang     = transcript_info.get("language", "en").upper()
    n_spk    = len(set(s["speaker"] for s in diarised_transcript))
    n_blocks = len(diarised_transcript)

    story.append(Paragraph("🎙️  MeetScribe — AI Meeting Intelligence Report", title_style))
    story.append(Paragraph(
        f"<b>Context:</b> {meeting_context}  &nbsp;|&nbsp;  "
        f"<b>Generated:</b> {now}  &nbsp;|&nbsp;  "
        f"<b>Duration:</b> {dur_min:.1f} min  &nbsp;|&nbsp;  "
        f"<b>Language:</b> {lang}  &nbsp;|&nbsp;  "
        f"<b>Speakers:</b> {n_spk}  &nbsp;|&nbsp;  "
        f"<b>Segments:</b> {n_blocks}",
        meta_style
    ))

    # ── Analytics Chart ───────────────────────────────────
    story.append(Paragraph("Meeting Analytics Dashboard", h2_style))
    if chart_path and os.path.exists(chart_path):
        story.append(RLImage(chart_path, width=6.5 * inch, height=2.1 * inch))
    else:
        story.append(Paragraph("(Chart not available)", meta_style))
    story.append(Spacer(1, 0.15 * inch))

    # ── Diarised Transcript ───────────────────────────────
    story.append(Paragraph("Diarised Transcript (Speaker-Attributed)", h2_style))

    max_blocks = min(len(diarised_transcript), 20)
    for seg in diarised_transcript[:max_blocks]:
        ts  = f"({seg['start']:.0f}s – {seg['end']:.0f}s)"
        txt = seg["text"].replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(
            f"<b>[{seg['speaker']}]</b>  "
            f"<font size=8 color='#888888'>{ts}</font>  {txt}",
            speaker_style
        ))

    if len(diarised_transcript) > max_blocks:
        story.append(Paragraph(
            f"<i>... {len(diarised_transcript) - max_blocks} more segments. "
            f"Full transcript in outputs/diarised_transcript.json</i>",
            meta_style
        ))
    story.append(Spacer(1, 0.15 * inch))

    # ── LLM Summaries ─────────────────────────────────────
    story.append(Paragraph("LLM Summarisation Comparison", h2_style))

    for model_name, result in llm_results.items():
        if result.get("status") != "success":
            story.append(Paragraph(
                f"<b>{model_name}</b>: {result.get('status', 'unavailable')}",
                meta_style
            ))
            continue

        cost_str = "FREE" if result["cost_usd"] == 0 else f"${result['cost_usd']:.5f}"
        story.append(Paragraph(
            f"<b>🤖  {result['model']}</b>  &nbsp;|&nbsp;  "
            f"⏱ {result['latency_sec']}s  &nbsp;|&nbsp;  "
            f"💰 {cost_str}  &nbsp;|&nbsp;  "
            f"🔤 {result['tokens']} tokens",
            body_style
        ))
        summary_safe = (
            result["summary"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        # Show up to 1000 chars of summary in PDF
        story.append(Paragraph(summary_safe[:1000], body_style))
        story.append(Spacer(1, 0.1 * inch))

    # ── Performance Comparison Table ──────────────────────
    story.append(Paragraph("Model Performance Summary", h2_style))

    table_data = [["Model", "Provider", "Latency", "Cost", "Tokens", "Status"]]
    for result in llm_results.values():
        cost_str = "FREE" if result.get("cost_usd", 0) == 0 else f"${result.get('cost_usd', 0):.5f}"
        table_data.append([
            result["model"][:25],
            result.get("provider", "—"),
            f"{result.get('latency_sec', '—')}s",
            cost_str,
            str(result.get("tokens", "—")),
            result.get("status", "—")[:15],
        ])

    col_widths = [2.0*inch, 1.0*inch, 0.8*inch, 0.9*inch, 0.7*inch, 1.1*inch]
    tbl = Table(table_data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1F3A8C")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        # All cells
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        # Alternating row colours
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EEF2FF")]),
        # Align numeric columns centre
        ("ALIGN",         (2, 0), (-1, -1), "CENTER"),
    ]))
    story.append(tbl)

    # ── Footer ────────────────────────────────────────────
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "<font color='#999999' size=8>"
        "Generated by MeetScribe | AlgoProfessor AI Internship 2026 | "
        "Pipeline: faster-whisper + pyannote.audio + Groq/Claude/GPT-4o + ReportLab"
        "</font>",
        body_style
    ))

    doc.build(story)
    print(f"✅ PDF report saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────
# 4. SLACK NOTIFICATION
# ─────────────────────────────────────────────────────────

def send_slack_notification(
    llm_results: dict,
    transcript_info: dict,
    slack_webhook_url: Optional[str] = None
) -> bool:
    """
    Send a meeting summary to a Slack channel via Incoming Webhook.

    Setup:
        1. api.slack.com/apps → Create New App → From Scratch
        2. Features → Incoming Webhooks → Activate
        3. Add New Webhook to Workspace → Copy URL
        4. Add to environment as SLACK_WEBHOOK_URL

    Args:
        llm_results       : LLM comparison results
        transcript_info   : Whisper transcript metadata
        slack_webhook_url : Slack Incoming Webhook URL

    Returns:
        True if notification sent successfully, False otherwise
    """
    import requests

    if not slack_webhook_url:
        print("⚠️  No SLACK_WEBHOOK_URL provided — skipping notification.")
        return False

    # Pick best available summary
    best_summary = "Summary not available."
    best_model   = "—"
    for preferred in ["Llama 3.3-70B (Groq FREE)", "Claude 3.5 Sonnet", "GPT-4o"]:
        r = llm_results.get(preferred, {})
        if r.get("status") == "success":
            best_summary = r["summary"][:500]
            best_model   = r["model"]
            break

    dur_min = transcript_info.get("duration", 0) / 60
    lang    = transcript_info.get("language", "en").upper()

    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🎙️ MeetScribe — New Meeting Processed"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Duration:* {dur_min:.1f} minutes"},
                    {"type": "mrkdwn", "text": f"*Language:* {lang}"},
                    {"type": "mrkdwn", "text": f"*Best Model Used:* {best_model}"},
                    {"type": "mrkdwn", "text": f"*Processed At:* {datetime.now().strftime('%H:%M')}"},
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary (by {best_model}):*\n{best_summary}..."
                }
            },
            {"type": "divider"}
        ]
    }

    try:
        response = requests.post(slack_webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Slack notification sent!")
            return True
        else:
            print(f"❌ Slack failed: HTTP {response.status_code} — {response.text}")
            return False
    except Exception as e:
        print(f"❌ Slack error: {e}")
        return False


# ─────────────────────────────────────────────────────────
# 5. QUICK USAGE EXAMPLE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Load outputs from previous pipeline stages
    transcript_path = "outputs/diarised_transcript.json"
    llm_path        = "outputs/llm_comparison.json"

    with open(transcript_path) as f:
        diarised = json.load(f)
    with open(llm_path) as f:
        results = json.load(f)

    # Minimal transcript_info for metadata
    transcript_info = {"duration": 300, "language": "en"}

    # Generate chart
    chart = generate_analytics_charts(diarised, results)

    # Generate PDF
    pdf = generate_pdf_report(
        diarised, results, transcript_info, chart,
        meeting_context="Business team standup"
    )

    print(f"\n📄 PDF ready: {pdf}")
