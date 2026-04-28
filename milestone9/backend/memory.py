"""
memory.py — Personal Knowledge Base Builder
--------------------------------------------
This file reads your profile data (from GitHub + manual input in profile.json)
and converts it into a single text "context" that is injected into every
AI prompt.

Think of it like giving the AI a cheat sheet about YOU before it answers
any recruiter question.

We also keep track of the conversation history (previous Q&A pairs) so the
AI has memory across multiple questions in the same session.
"""

import json
import os
from typing import Optional


def load_profile(profile_path: str = "data/profile.json") -> Optional[dict]:
    """
    Load the saved profile JSON file.
    
    Returns the dict or None if file doesn't exist yet.
    """
    if not os.path.exists(profile_path):
        return None
    
    with open(profile_path, "r") as f:
        return json.load(f)


def build_context_text(profile: dict) -> str:
    """
    Convert the profile dict into a well-formatted text block.
    This text block is injected at the top of every AI prompt as "context".
    
    The better this text is formatted, the better the AI's answers will be.
    This is called "prompt engineering" — crafting good input for better output.
    
    Example output (trimmed):
    ---
    NAME: Rahul Sharma
    LOCATION: Hyderabad, India
    BIO: AI/ML Engineer passionate about building real-world AI systems
    SKILLS/LANGUAGES: Python, JavaScript, C++
    ...
    ---
    """
    p = profile.get("profile", {})
    repos = profile.get("repositories", [])
    languages = profile.get("languages", [])
    
    lines = []
    
    # ── Basic Identity ──
    lines.append("=== CANDIDATE PROFILE ===")
    lines.append(f"NAME: {p.get('name', 'Unknown')}")
    lines.append(f"LOCATION: {p.get('location', 'Not specified')}")
    lines.append(f"BIO: {p.get('bio', 'Not specified')}")
    
    if p.get("company"):
        lines.append(f"CURRENT COMPANY/INSTITUTION: {p['company']}")
    
    if p.get("blog"):
        lines.append(f"PORTFOLIO/WEBSITE: {p['blog']}")
    
    lines.append(f"GITHUB: {p.get('github_url', '')}")
    lines.append(f"GITHUB STATS: {p.get('public_repos', 0)} public repos, {p.get('followers', 0)} followers")
    lines.append("")
    
    # ── Technical Skills ──
    lines.append("=== TECHNICAL SKILLS ===")
    if languages:
        lines.append(f"PROGRAMMING LANGUAGES: {', '.join(languages)}")
    
    # Include manually added skills from profile.json (if any)
    extra = profile.get("extra_info", {})
    if extra.get("skills"):
        lines.append(f"ADDITIONAL SKILLS: {', '.join(extra['skills'])}")
    if extra.get("frameworks"):
        lines.append(f"FRAMEWORKS/TOOLS: {', '.join(extra['frameworks'])}")
    lines.append("")
    
    # ── Projects ──
    lines.append("=== GITHUB PROJECTS ===")
    for i, repo in enumerate(repos[:10], 1):   # Top 10 repos
        lines.append(f"{i}. {repo['name'].upper()}")
        lines.append(f"   Description: {repo['description']}")
        lines.append(f"   Language: {repo['language']}")
        lines.append(f"   Stars: ⭐ {repo['stars']} | Forks: 🍴 {repo['forks']}")
        if repo.get("topics"):
            lines.append(f"   Topics: {', '.join(repo['topics'])}")
        lines.append(f"   URL: {repo['url']}")
        lines.append("")
    
    # ── Extra Info (manually provided in profile.json) ──
    if extra:
        lines.append("=== ADDITIONAL INFORMATION ===")
        if extra.get("education"):
            lines.append(f"EDUCATION: {extra['education']}")
        if extra.get("experience"):
            lines.append(f"EXPERIENCE: {extra['experience']}")
        if extra.get("achievements"):
            lines.append(f"ACHIEVEMENTS: {extra['achievements']}")
        if extra.get("certifications"):
            lines.append(f"CERTIFICATIONS: {extra['certifications']}")
        if extra.get("linkedin"):
            lines.append(f"LINKEDIN: {extra['linkedin']}")
        lines.append("")
    
    lines.append("=== END OF PROFILE ===")
    
    return "\n".join(lines)


def format_conversation_history(history: list[dict]) -> str:
    """
    Convert previous Q&A pairs into a readable text block for the AI.
    
    History format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    
    We keep only the last 6 messages (3 turns) to avoid making the prompt too long.
    """
    if not history:
        return ""
    
    # Keep last 6 messages (3 user + 3 assistant turns)
    recent = history[-6:]
    
    lines = ["=== CONVERSATION HISTORY ==="]
    for msg in recent:
        role = "Recruiter" if msg["role"] == "user" else "You (AI Agent)"
        lines.append(f"{role}: {msg['content']}")
    lines.append("=== END OF HISTORY ===\n")
    
    return "\n".join(lines)
