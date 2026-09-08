# NexusFlow — AI Traffic Intelligence System

A smart-city traffic control platform combining YOLOv8 vehicle detection, a Deep Q-Network for adaptive signal control, and a SUMO-based simulation environment for training — with a Next.js dashboard for live monitoring, video analysis, and a training console.

---

## What it does

- **Vehicle detection** — uploads a traffic video, runs it through **YOLOv8** frame-by-frame, and returns per-lane vehicle counts, bounding boxes, and a traffic heatmap
- **Signal optimization** — a **Deep Q-Network (PyTorch)** trained against a custom 10-dimensional state (per-lane queue length, per-lane wait time, current phase, time in phase) to decide whether to hold or switch a traffic signal, with a reward function that balances reduced wait time against unnecessary switching
- **Emergency vehicle priority** — detects an emergency vehicle in a lane and overrides the signal decision to clear that direction
- **SUMO simulation environment** — a full TraCI-based training environment (custom intersection network, traffic generator, reward shaping) used to train the DQN offline before deployment
- **Live dashboard** — Next.js frontend with a digital-twin intersection view, analytics (reward/loss/epsilon curves), video analysis panel, and a training console

## Tech stack

**Backend:** FastAPI, PyTorch (DQN), Ultralytics YOLOv8, OpenCV, SUMO + TraCI, WebSockets (live streaming), NumPy/Pandas

**Frontend:** Next.js 14, React 18, TypeScript, Tailwind CSS, Recharts

## Architecture

```
backend/
├── app/main.py                    # FastAPI app, router registration, startup checks
├── app/api/
│   ├── traffic_routes.py          # /api/traffic — signal optimization + AI explanations
│   ├── training_routes.py         # /api/video — video upload + YOLO detection
│   ├── video_routes.py            # /api/training — training console (see note below)
│   └── websocket_routes.py        # live streaming to the dashboard
├── app/models/dqn_agent.py        # DQN agent used by the live /api/traffic endpoint
├── app/rl/                        # DQN agent + training manager used for offline SUMO training
├── app/sumo/                      # TraCI environment, traffic generator, state manager
├── app/vision/                    # YOLOv8 detector, tracker, lane counter, heatmap
├── app/services/video_detector.py # YOLOv8 service backing the video-upload endpoint
└── models/dqn_checkpoint.pth      # trained model weights from an offline SUMO training run

frontend/
└── src/app/                       # dashboard, simulation (digital twin), video-analysis,
                                    # training-console, analytics, emergency, architecture pages
```

Note: `training_routes.py` and `video_routes.py` are named opposite to what they contain — the video-upload/YOLO endpoints live in `training_routes.py`, and the training-console endpoints live in `video_routes.py`. Worth a rename.

## Honest status of the live demo

The DQN, SUMO environment, and YOLO detector are all real, working implementations, and a real offline training run produced the saved checkpoint (`dqn_checkpoint.pth`). Two parts of the **live API** don't currently reflect that:

- **`/api/traffic/optimize-signal`** tries to import `DQNAgent` from `app/rl/dqn_agent.py`, but that module only defines `EnhancedDQNAgent` — so the import fails and the endpoint silently falls back to a simple rule (send green to whichever of N/S or E/W has the larger vehicle count). The response still labels this `"policy": "DQN Reinforcement Learning"`, which is misleading; that needs fixing so it either loads the real trained agent or reports honestly that it's on the fallback.
- **`/api/training/start`** (the "live training console") generates its reward/loss/epsilon curves with `random.uniform()` and a decay formula — it doesn't run the real `DQNAgent`/`SumoEnvironment`/replay buffer loop. It's a demo of what a training curve looks like, not real training telemetry. Actual training happens by running the RL/SUMO code directly (see below), not through this endpoint.

## Running locally

**Backend**
```bash
cd backend
pip install -r requirements.txt
python run.py
```

**Offline DQN training (real, against SUMO)**
```bash
cd backend
python -m app.sumo.environment   # requires SUMO installed and on PATH
# or use app/rl/training_manager.py to run a full training loop
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

API docs at `/docs` once the backend is running.
