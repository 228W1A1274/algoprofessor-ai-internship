"""
github_loader.py — GitHub Profile Data Extractor
-------------------------------------------------
FIXED: Handles None values for description, language, topics, etc.
GitHub API returns null (None in Python) for repos with no description or language set.
Using `or` instead of `.get(key, default)` properly handles None values.
"""

import requests
import json
from config import GITHUB_USERNAME, GITHUB_TOKEN


def get_github_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def fetch_user_profile(username: str) -> dict:
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url, headers=get_github_headers(), timeout=10)
    response.raise_for_status()
    data = response.json()

    return {
        "name":         data.get("name") or username,
        "bio":          data.get("bio") or "",
        "location":     data.get("location") or "",
        "email":        data.get("email") or "",
        "blog":         data.get("blog") or "",
        "company":      data.get("company") or "",
        "public_repos": data.get("public_repos") or 0,
        "followers":    data.get("followers") or 0,
        "following":    data.get("following") or 0,
        "github_url":   data.get("html_url") or "",
        "avatar_url":   data.get("avatar_url") or "",
        "joined":       (data.get("created_at") or "")[:10],
    }


def fetch_repositories(username: str, max_repos: int = 20) -> list:
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        "sort": "updated",
        "per_page": max_repos,
        "type": "owner"
    }

    response = requests.get(url, headers=get_github_headers(), params=params, timeout=10)
    response.raise_for_status()

    repos = []
    for repo in response.json():
        if repo.get("fork"):
            continue

        # FIX: Use `or` instead of .get(key, default)
        # Reason: GitHub returns null (None) for description/language on empty repos.
        # .get("description", "No description") does NOT handle None — it returns None!
        # But `repo.get("description") or "No description"` correctly falls back.
        repos.append({
            "name":        repo.get("name") or "unnamed",
            "description": repo.get("description") or "No description",
            "language":    repo.get("language") or "Unknown",
            "stars":       repo.get("stargazers_count") or 0,
            "forks":       repo.get("forks_count") or 0,
            "topics":      repo.get("topics") or [],
            "url":         repo.get("html_url") or "",
            "updated":     (repo.get("updated_at") or "")[:10],
        })

    return repos


def build_github_summary(username: str) -> dict:
    print(f"📡 Fetching GitHub profile for: {username}")

    profile = fetch_user_profile(username)
    print(f"  ✅ Got profile: {profile['name']}")

    repos = fetch_repositories(username)
    print(f"  ✅ Got {len(repos)} repositories")

    # FIX: Filter out None and "Unknown" before sorting
    languages = list(set(
        repo["language"] for repo in repos
        if repo["language"] and repo["language"] != "Unknown"
    ))

    # FIX: Sort safely — all values guaranteed to be strings now
    languages_sorted = sorted(languages)

    # Find most starred repo safely
    top_repo = max(repos, key=lambda r: r["stars"] or 0) if repos else None

    return {
        "profile":      profile,
        "repositories": repos,
        "languages":    languages_sorted,
        "top_project":  top_repo,
        "total_repos":  len(repos),
    }


def save_github_data(username: str, output_path: str = "data/profile.json") -> dict:
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = build_github_summary(username)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  💾 Saved to {output_path}")
    return data