# Deploy to Render (Streamlit + FastAPI)

Two free-tier services: one for the Streamlit UI, one for the FastAPI `/recommend` endpoint.

## Prereqs
- Render account
- Repository pushed to GitHub (or Render-connected Git)
- Models/data bundled or retrained at runtime (see “Model assets” below)

## Steps
1) Commit `render.yaml` and push to your repo.
2) In Render, create a “Blueprint” and point it at your repo. Render will detect `render.yaml` and create two services:
   - `nfl-gm-streamlit` running `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
   - `nfl-gm-api` running `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
3) Set env vars in both services as needed (`PYTHON_VERSION` already set in the blueprint; add API keys if you wire LLMs).

## Model assets
- Easiest: check in a pre-trained `models/wp_model.joblib` and `models/rates/*.joblib`, or host them in object storage and download at startup.
- If you need to train on-deploy, add a build/exec hook that runs the pipeline (beware free-tier time limits).

## Testing the deploy
- Streamlit: open the `nfl-gm-streamlit` URL; load a preset, verify recommendation renders.
- API: `curl -X POST https://<api-host>/recommend -H "Content-Type: application/json" -d '{"home_team":"KC","away_team":"BUF","posteam":"KC","yardline_100":50,"down":4,"ydstogo":1,"home_score":14,"away_score":10,"game_seconds_remaining":600,"home_timeouts":3,"away_timeouts":3}'`

## Capturing a short walkthrough
- Run the deployed Streamlit URL in your browser.
- Record a 20–30s clip (e.g., QuickTime screen recording or Loom) showing:
  1) Selecting a preset
  2) Recommendation + WP delta vs actual
  3) Note about learned rates vs heuristics
- Save the clip and link it in your README or LinkedIn post.
