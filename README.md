# EMG-AI-Arm

**Multimodal Sensor Fusion for Intelligent Robotic Arm Control Using Electromyography and Computer Vision**

Undergraduate thesis project — Department of Computer Science & Engineering,
State University of Bangladesh.

A dual-modality gesture-control system that drives a simulated 6-degree-of-freedom robotic arm from either a webcam (via MediaPipe hand-landmark detection) or an 8-channel surface EMG signal, unified behind a single shared command interface. The EMG path is trained and validated on the public "EMG data for Gestures" dataset, with live inference validated end-to-end on a subject never seen during training. Ships with a PyQt6 GUI for live signal monitoring, cue-based training, and inference with a live 3D arm preview.

---

## Status

| Component | State |
|---|---|
| Camera-based gesture recognition (MediaPipe + Random Forest) | Working |
| EMG-based gesture recognition (public dataset, 8-channel) | Working |
| Feature extraction (80 time-domain features: 10/channel × 8 channels) | Working |
| Random Forest (deployed EMG model) | Working |
| Comparison baselines (RF variants, XGBoost, LightGBM, 1D CNN) | Working — none outperformed deployed RF |
| Live inference pipeline (held-out-subject validated) | Working |
| Simulated 6-DOF arm (3D visualization) | Working |
| PyQt6 GUI (Live Signal / Trainer / Inference + arm) | Working |
| Custom analog EMG hardware acquisition | Not part of final system — see Limitations |

## Architecture

```
        Camera (webcam)                    EMG (public dataset / replay)
             │                                        │
             ▼                                        ▼
    MediaPipe hand landmarks              Windowing (200 samples)
             │                                        │
             ▼                                        ▼
    Random Forest (camera)              80-feature extraction (8ch × 10)
             │                                        │
             ▼                                        ▼
                    ┌─────────────────────────┐
                    │   RobotCommand (shared) │
                    └─────────────────────────┘
                                 │
                                 ▼
                  Simulated 6-DOF arm + PyQt6 GUI
```

Both pipelines are implemented independently and neither is aware the other exists — they are unified only at the shared `RobotCommand` interface.

## Repository layout

```
emg_ai_arm/
├── acquisition/       EMG replay from CSV (replay_csv_8ch.py)
├── camera/            MediaPipe hand tracking, landmarks, camera classifier
├── control/           robot_command.py (shared interface), gesture_mapper.py
├── emulator/          Synthetic EMG generator (fake_emg_8ch.py)
├── features/          extract_features_8ch.py — 80-feature time-domain extraction
├── models/            Trained classifiers (gitignored — see below)
├── visualization/     PyQt6 app, arm_widget, inference_tab, signal_tab, stream_worker
└── data/              raw/ and windows/ (gitignored)
paper/                 LaTeX manuscript (IEEEtran)
docs/                  Diagrams, schematics, design notes
```

<!-- TODO: verify this against the real tree before committing — run
     `git ls-files emg_ai_arm/ | sort` and confirm every folder/file above
     actually exists and nothing current is missing. -->

## Quickstart

```bash
# 1. Clone
git clone https://github.com/Purn0/emg-ai-arm.git
cd emg-ai-arm

# 2. Create environment
python -m venv .venv
.\.venv\Scripts\activate         # Windows
# source .venv/bin/activate      # Linux/macOS

# 3. Install
pip install -r requirements.txt

# 4. Generate a synthetic EMG stream (no hardware / dataset needed)
python -m emg_ai_arm.emulator.fake_emg_8ch

# 5. Train the EMG baseline
python -m emg_ai_arm.features.extract_features_8ch
# TODO: confirm the actual training entry point/script name — this may
# not be the real command. Check what you currently run to train the
# deployed Random Forest model and replace this line.

# 6. Launch the GUI
python -m emg_ai_arm.visualization.app
```

<!-- TODO: steps 4–6 are reconstructed from repository file names, not
     verified against your actual current CLI. Please confirm each
     command runs before committing this file. -->

## Hardware (attempted, not part of final delivered system)

An initial single-channel surface EMG sensor was prototyped for live acquisition but did not yield sufficiently reliable signal within the project timeline. The EMG classification component in the delivered system instead uses the public multi-subject dataset described below. The table reflects the original hardware plan, kept here for reference.

| Part | Qty | Approx. price (BDT) |
|---|---|---|
| AD620 instrumentation amplifier | 3 | 1500 |
| TL072 op-amp (filter stages) | 2 | 100 |
| Ag/AgCl ECG electrodes (disposable, pack) | 1 | 300 |
| Arduino Nano | 1 | 400 |
| 9 V battery + clip, perfboard, passives | — | 300 |
| **Total** | | **~2600 BDT** |

## Data

The EMG classification component uses the publicly available **"EMG data for Gestures"** dataset — eight-channel surface EMG recordings from a MYO Thalmic bracelet across 36 subjects. If you use this repository's EMG pipeline, please also cite the original dataset:

> Krilova, N., Kastalskiy, I., Kazantsev, V., Makarov, V. A., & Lobov, S. (2018). *EMG data for Gestures* [Dataset]. UCI Machine Learning Repository.

The camera-based gesture dataset was self-collected specifically for this project.

## Citing

If this code or system is useful in your work, please cite:

> Ahsan, M. A. (2026). *Multimodal Sensor Fusion for Intelligent Robotic Arm Control Using Electromyography and Computer Vision.* Undergraduate Thesis, Department of CSE, State University of Bangladesh.

## License

MIT — see [LICENSE](LICENSE).
