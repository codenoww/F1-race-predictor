# Legacy pipeline (not used)

The original predictor (`webapp/`) and the PC scheduled-task script (`scripts/`) relied on F1's
live-timing servers through FastF1 for lap, tyre, weather and practice features. F1 now answers
HTTP 401 to everyone, so these can no longer produce new predictions. They are kept for reference.

The live system is in `dashboard/` (see `.github/workflows/update-dashboard.yml`).
