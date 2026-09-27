// Triggers the "Update F1 prediction" GitHub Actions workflow on demand.
// Requires a GH_DISPATCH_TOKEN env var set in Vercel's dashboard (Project
// Settings -> Environment Variables) — a fine-grained GitHub PAT scoped to
// this repo only, with "Actions: read and write" permission. Never hardcode
// it here; it's read from the server-side environment at request time and
// never sent to the browser.

const OWNER = "codenoww";
const REPO = "F1-race-predictor";
const WORKFLOW_FILE = "update-prediction.yml";

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Use POST" });
  }

  const token = process.env.GH_DISPATCH_TOKEN;
  if (!token) {
    return res.status(500).json({
      error: "Server isn't configured yet — GH_DISPATCH_TOKEN is missing. " +
             "Add it in Vercel Project Settings > Environment Variables.",
    });
  }

  try {
    const ghRes = await fetch(
      `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ref: "main" }),
      }
    );

    if (ghRes.status === 204) {
      return res.status(200).json({ ok: true, message: "Workflow run triggered." });
    }

    const text = await ghRes.text();
    return res.status(502).json({
      error: `GitHub API returned ${ghRes.status}`,
      detail: text,
    });
  } catch (e) {
    return res.status(500).json({ error: String(e) });
  }
}
