# EMG-AI-Arm

**Intelligent sEMG-Based Human–Machine Interface for Multi-DoF Robotic Arm Control**

Undergraduate thesis project — Department of Computer Science & Engineering,
State University of Bangladesh.

A low-cost surface electromyography (sEMG) pipeline that decodes muscle intent in real time and drives a 6-DoF robotic arm. The system targets sub-$25 hardware, runs on commodity laptops, and ships with a PyQt6 GUI for cue-based training, live signal monitoring, and inference with a virtual arm preview.

---

## Status

| Component | State |
|---|---|
| Software pipeline (emulator-based) | Working |
| Preprocessing (filters, envelope, windowing) | Working |
| Feature extraction (time + frequency domain) | Working |
| Random Forest baseline | Working |
| 1D CNN baseline | Working |
| State-machine controller | Working |
| PyQt6 GUI (signal / training / inference) | In progress |
| Serial acquisition from Arduino | Stub (hardware pending) |
| Real EMG dataset | Pending hardware |

## Architecture

```
Muscle activity
    │
    ▼  surface electrodes (Ag/AgCl)
Analog front-end (AD620 + active filters, 20–450 Hz, 50 Hz notch)
    │
    ▼  ADC, 1 kHz raw / 200 Hz envelope
Arduino Nano (USB serial @ 115200 baud)
    │
    ▼
Python pipeline
    ├── acquisition/   serial_reader OR fake_emg emulator
    ├── preprocessing/ filters, envelope, windowing
    ├── features/      RMS, MAV, WL, ZC, SSC, MNF, MDF
    ├── models/        RF, SVM, 1D-CNN
    ├── control/       mode-based state machine
    └── visualization/ PyQt6 GUI + 2D arm preview
    │
    ▼
6-DoF robotic arm (6× servos via Arduino PWM)
```

## Repository layout

```
emg_ai_arm/
├── acquisition/      Serial reader for live hardware
├── emulator/         Synthetic EMG generator for development without hardware
├── preprocessing/    Bandpass, notch, rectification, envelope, windowing
├── features/         Time- and frequency-domain feature extractors
├── models/           train_ml.py (RF), train_cnn.py (PyTorch CNN), inference.py
├── control/          State machine mapping classifier output to arm DOFs
├── visualization/    PyQt6 GUI app + live plots + 2D arm widget
├── utils/            Project paths and config
└── data/
    ├── raw/          Raw recordings (gitignored)
    └── windows/      Windowed datasets (gitignored)
paper/                LaTeX manuscript (IEEEtran)
docs/                 Diagrams, schematics, design notes
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-username>/emg-ai-arm.git
cd emg-ai-arm

# 2. Create environment
python -m venv .venv
.\.venv\Scripts\activate         # Windows
# source .venv/bin/activate      # Linux/macOS

# 3. Install
pip install -r requirements.txt

# 4. Generate synthetic dataset
python -m emg_ai_arm.emulator.fake_emg

# 5. Train baseline
python -m emg_ai_arm.models.train_ml

# 6. Launch GUI
python -m emg_ai_arm.visualization.app
```

## Hardware (planned)

| Part | Qty | Approx. price (BDT) |
|---|---|---|
| AD620 instrumentation amplifier | 3 | 1500 |
| TL072 op-amp (filter stages) | 2 | 100 |
| Ag/AgCl ECG electrodes (disposable, pack) | 1 | 300 |
| Arduino Nano | 1 | 400 |
| 9 V battery + clip, perfboard, passives | — | 300 |
| **Total** | | **~2600 BDT** |

## Citing

If this code or system is useful in your work, please cite:

> Ahsan, M. A. (2026). *Design and Implementation of an Intelligent sEMG-Based Human–Machine Interface for Multi-DoF Robotic Arm Control.* Undergraduate Thesis, Department of CSE, State University of Bangladesh.

## License

MIT — see [LICENSE](LICENSE).
