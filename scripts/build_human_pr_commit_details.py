import os, time, json
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
RQ1_MAIN_FRAME = os.path.join(BASE_DIR, "results", "rq1_main_frame.parquet")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "human_pr_commit_details.parquet")
DONE_PATH = os.path.join(BASE_DIR, "data", "processed", "human_pr_commit_details_done.json")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # put in .env
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

def gh_get(url):
    r = requests.get(url, headers=HEADERS)
    # basic rate-limit handling
    if r.status_code == 403 and "rate limit" in r.text.lower():
        reset = int(r.headers.get("X-RateLimit-Reset", time.time()+60))
        sleep_for = max(5, reset - int(time.time()) + 5)
        print(f"Rate limited. Sleeping {sleep_for}s")
        time.sleep(sleep_for)
        return gh_get(url)
    r.raise_for_status()
    return r

def load_done():
    if os.path.exists(DONE_PATH):
        with open(DONE_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_done(done_set):
    with open(DONE_PATH, "w") as f:
        json.dump(sorted(list(done_set)), f)

def main(limit_prs=None):
    df = pd.read_parquet(RQ1_MAIN_FRAME, columns=["id","ai_pr","repo_full_name","number"])
    human_prs = df[df.ai_pr == 0].copy()

    done = load_done()
    rows = []

    for i, pr in human_prs.iterrows():
        pr_id = int(pr["id"])
        if str(pr_id) in done:
            continue

        owner, repo = pr["repo_full_name"].split("/")
        number = int(pr["number"])

        # 1) list commits in PR
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/commits?per_page=100"
        commits = gh_get(commits_url).json()

        for c in commits:
            sha = c["sha"]

            # 2) commit details -> files with patches
            commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
            commit_obj = gh_get(commit_url).json()

            for fobj in commit_obj.get("files", []):
                patch = fobj.get("patch")
                if not patch:
                    continue
                rows.append({
                    "pr_id": pr_id,
                    "sha": sha,
                    "filename": fobj.get("filename", ""),
                    "patch": patch
                })

        done.add(str(pr_id))

        # periodic flush
        if len(done) % 50 == 0:
            print(f"Flushing at {len(done)} PRs, rows={len(rows)}")
            out_df = pd.DataFrame(rows)
            if os.path.exists(OUT_PATH):
                prev = pd.read_parquet(OUT_PATH)
                out_df = pd.concat([prev, out_df], ignore_index=True)
            out_df.to_parquet(OUT_PATH, index=False)
            rows = []
            save_done(done)

        if limit_prs and len(done) >= limit_prs:
            break

    # final flush
    if rows:
        out_df = pd.DataFrame(rows)
        if os.path.exists(OUT_PATH):
            prev = pd.read_parquet(OUT_PATH)
            out_df = pd.concat([prev, out_df], ignore_index=True)
        out_df.to_parquet(OUT_PATH, index=False)

    save_done(done)
    print("Done. PRs processed:", len(done))

if __name__ == "__main__":
    # main(limit_prs=200)  # start small
    main(limit_prs=None)
