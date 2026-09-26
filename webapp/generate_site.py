"""
Regenerates public/index.html with a fresh prediction for whatever race is next.
Run by the GitHub Actions workflow on a schedule (see .github/workflows/update-prediction.yml).
"""
import json
import os
import re

import predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
TEMPLATE_PATH = os.path.join(BASE_DIR, "site_template.html")
OUTPUT_PATH = os.path.join(REPO_ROOT, "public", "index.html")


def main():
    result = predictor.run_prediction(top_n=10, force=True)

    template = open(TEMPLATE_PATH, encoding="utf-8").read()
    data = json.dumps(result)

    pattern = re.compile(r"const PREDICTION = \{.*?\};\s*//\s*END_PREDICTION", re.DOTALL)
    replacement = f"const PREDICTION = {data}; // END_PREDICTION"
    new_html, n = pattern.subn(replacement, template, count=1)
    if n != 1:
        raise RuntimeError("template placeholder not found or matched more than once")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    open(OUTPUT_PATH, "w", encoding="utf-8").write(new_html)
    print(f"Generated {OUTPUT_PATH}")
    print(f"Race: {result['race_name']} (Round {result['round']}, {result['season']})")
    print("Predicted top 3:", [r["driver"] for r in result["ranked"][:3]])


if __name__ == "__main__":
    main()
