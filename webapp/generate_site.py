"""
Regenerates the published prediction for whatever race is next.

Writes two files into public/:
  - prediction.json : fetched by the live page at runtime straight from GitHub,
                      so a new prediction shows up without redeploying the site
  - index.html      : same data embedded as an offline/first-paint fallback

Run by the GitHub Actions workflow (see .github/workflows/update-prediction.yml).
"""
import json
import os
import re
from datetime import datetime, timezone

import predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
TEMPLATE_PATH = os.path.join(BASE_DIR, "site_template.html")
PUBLIC_DIR = os.path.join(REPO_ROOT, "public")
OUTPUT_HTML = os.path.join(PUBLIC_DIR, "index.html")
OUTPUT_JSON = os.path.join(PUBLIC_DIR, "prediction.json")


def main():
    result = predictor.run_prediction(top_n=10, force=True)

    if not result.get("ranked"):
        # The next race has no quali/practice data yet (or the fetch was
        # blocked/rate-limited). Leave whatever is already published alone
        # rather than overwrite a good prediction with an empty podium.
        print(
            f"No quali/practice data yet for {result['race_name']} "
            f"(Round {result['round']}, {result['season']}) - skipping, "
            f"leaving existing output untouched."
        )
        for line in result.get("log", [])[-3:]:
            print("  log:", line)
        return

    result["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = json.dumps(result)

    template = open(TEMPLATE_PATH, encoding="utf-8").read()
    pattern = re.compile(r"let PREDICTION = \{.*?\};\s*//\s*END_PREDICTION", re.DOTALL)
    replacement = f"let PREDICTION = {data}; // END_PREDICTION"
    new_html, n = pattern.subn(lambda _m: replacement, template, count=1)
    if n != 1:
        raise RuntimeError("template placeholder not found or matched more than once")

    os.makedirs(PUBLIC_DIR, exist_ok=True)
    open(OUTPUT_HTML, "w", encoding="utf-8").write(new_html)
    open(OUTPUT_JSON, "w", encoding="utf-8").write(data)
    print(f"Generated {OUTPUT_HTML} and {OUTPUT_JSON}")
    print(f"Race: {result['race_name']} (Round {result['round']}, {result['season']})")
    print("Predicted top 3:", [r["driver"] for r in result["ranked"][:3]])


if __name__ == "__main__":
    main()
