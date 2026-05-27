# AGENTS.md

## Primary workflow

- The main supported workflow is Dockerized Jupyter Lab:
  `docker compose up --build jupyter`
- Open `playground/gradcam_inference.ipynb` in Jupyter Lab and run the notebook there.
- The container command intentionally mirrors the local command:
  `uv run jupyter lab --no-browser --ip=0.0.0.0 --port=8888`.
- Keep the host port bind `8888:8888`; this machine is intended to be reachable directly from the DMZ/LAN as well as through Cloudflare Tunnel.

## Dataset and model mounts

- Do not commit datasets, checkpoints, or other large model artifacts.
- `playground/gradcam_inference.ipynb` uses these clean, hardcoded container paths:
  - Dataset: `/data/lampe/dataset`
  - Model: `/data/lampe/model_v2.pth`
- `compose.yaml` bind-mounts host paths to those container paths. Keep host paths out of Python/notebook code. If host paths differ, copy `.env.example` to `.env` and change:
  - `LAMPE_HOST_DATASET_DIR`
  - `LAMPE_HOST_MODEL_PATH`

## Cloudflare Tunnel

- `compose.yaml` includes an optional `cloudflared` sidecar under the `tunnel` profile.
- To expose Jupyter through Cloudflare, configure a public hostname in Cloudflare Zero Trust that routes to `http://jupyter:8888`.
- Store the tunnel token only in local `.env` as `CLOUDFLARE_TUNNEL_TOKEN`; never commit real tunnel tokens.
- Start the tunneled stack with `docker compose --profile tunnel up --build`.
- Keep Cloudflare Access and a strong `JUPYTER_TOKEN` enabled for internet-facing deployments.

## Development notes

- Keep Python dependency changes in `pyproject.toml`/`uv.lock`; use `uv add <package>` rather than editing lock files manually.
- Keep the Docker virtual environment outside the repository via `UV_PROJECT_ENVIRONMENT=/opt/lampe-venv`; this avoids using a host `.venv` inside Linux containers.
- `src/shared/constants.py` selects `cuda`, then `mps`, then `cpu`; do not hardcode `DEVICE = "mps"` because Docker Linux containers need a CPU/CUDA fallback.
