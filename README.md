# EMG-AI-Arm

Gesture control of a simulated six-joint robotic arm from two independent
inputs: a webcam (MediaPipe hand landmarks, eight gestures) and eight-channel
surface EMG (seven classes of the public UCI "EMG data for Gestures" set). Both
inputs end in the same `RobotCommand` type, so the arm and its control code do
not depend on where a command came from. Everything runs on a laptop CPU.

Undergraduate thesis project, Department of CSE, State University of
Bangladesh (2026): *Multimodal Sensor Fusion for Intelligent Robotic Arm
Control Using Electromyography and Computer Vision*. The two inputs are used
one at a time; fusing their evidence is future work. The code as defended is
tagged [`thesis-2026`](https://github.com/Purn0/emg-ai-arm/tree/thesis-2026); `main` has been cleaned up and
improved since (see [Changes since the thesis](#changes-since-the-thesis)).

The camera recognizer and its landmark dataset come from
[gesture-control-project](https://github.com/Purn0/gesture-control-project).

## Results

All EMG numbers come from six-fold subject-wise cross-validation over the 36
subjects: each subject is tested once, by a model trained on 30 other
subjects. Mean +- standard deviation over subjects.

| EMG pipeline (Random Forest) | 7 classes, "unmarked" included: accuracy / macro-F1 | 6 labelled gestures: accuracy / macro-F1 |
|---|---|---|
| Thesis features, balanced class weights | 56.8 +- 6.3 % / 0.496 | 82.6 +- 12.9 % / 0.816 |
| TD9 features + per-user calibration (app model) | 62.7 +- 5.9 % / 0.541 | 87.4 +- 12.5 % / 0.870 |

The thesis recipe itself (class 0 downsampled to 4,000 windows) gives
47.7 +- 6.7 % / 0.488 on 7 classes. Replaying subject 05, whom neither model
saw in training, through the app: the thesis model reaches 52.6 % (macro-F1
0.518), the app model 62.8 % (0.605). Every configuration, and paired
significance tests, are in
[experiments/results/summary.md](experiments/results/summary.md).

What matters most:
- **The "unmarked" class.** It covers 64 % of the windows, and most of it is
  the relaxed hand between gestures, which looks like "hand at rest". Without
  it, the same pipeline recognises the six labelled gestures far better.
- **Per-user calibration.** Standardising each feature with the mean and
  standard deviation of a recording of the same person going through the
  gestures once (no labels needed) removes much of the person-to-person
  difference in signal amplitude.
- Feature sets designed for low-rate armbands (Phinyomark et al.'s TD4/TD9),
  gradient boosting and a 1D CNN change the results far less.

In the dataset, a subject's two sessions were recorded 1-3 minutes apart
(going by the times in the file names), probably without taking the bracelet
off, so calibrating from the other session is the favourable same-sitting
case; it does not test putting the bracelet back on or recording on another
day.

Camera: 96.9 % on 32 held-out capture bursts, 97.0 +- 2.5 % in 5x5
burst-grouped cross-validation (see gesture-control-project).

## Quick start

Python 3.11 (the versions in `requirements.txt` are the tested ones).

```
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m experiments.download_data    # UCI dataset, 17 MB, CC BY 4.0
python -m experiments.train_models     # EMG model (a few minutes) + camera model
python -m emg_ai_arm.visualization.app
```

Pick a source and press **Start**:

- **Replay: recorded EMG (subject 05)** streams a recorded session through the
  whole pipeline (windowing, features, calibration from the subject's other
  session, classification, command, arm) and shows a running score against
  the recorded labels. It runs at 200 samples/s, about five times slower than
  the recording, so the arm motion is easy to follow.
- **Synthetic EMG** is a made-up signal for checking that the GUI and threads
  run. It is not EMG, and the models do not classify it meaningfully.
- **Camera** needs a webcam. Predictions below 0.6 confidence are ignored.

There is no live EMG sensor input: a single-channel sensor tried during the
thesis did not give reliable signals.

## How the EMG pipeline works

1. **Windows**: 200 samples of all 8 channels (about 0.2 s of recording),
   labelled by their first sample.
2. **Features**: `emg_ai_arm/features/emg_features.py`. The app uses the TD9
   set of Phinyomark et al. (2018): L-scale, maximum fractal length, mean
   square root, Willison amplitude, zero crossings, RMS, integrated absolute
   value, DASDV and variance, per channel (72 values). The thesis set (mean,
   SD, RMS, energy, min, max, range, median, MAV, sign changes) is also there.
3. **Calibration**: features are standardised with the mean and SD of a
   recording in which the same person performs the gestures once; its labels
   are not used (`emg_ai_arm/emg_model.py`). For the replay, that recording
   is subject 05's other session.
4. **Classifier**: Random Forest, 300 trees, balanced class weights, trained
   on all subjects except 05.
5. **Command**: class 0-6 -> `RobotCommand` 0-6 (the thesis mapping):

| EMG class | Command | Camera gesture |
|---|---|---|
| 0 unmarked | STOP | - |
| 1 hand at rest | ARM_UP | Thumb up |
| 2 fist | ARM_DOWN | Thumb down |
| 3 wrist flexion | GRIP_OPEN | Open palm |
| 4 wrist extension | GRIP_CLOSE | Closed fist |
| 5 radial deviation | WRIST_CW | Rock |
| 6 ulnar deviation | WRIST_CCW | Call me |
| - | BASE_LEFT | Victory |
| - | BASE_RIGHT | Pointing up |

## Reproducing the experiments

```
pip install -r requirements-experiments.txt
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # CPU build
python -m experiments.benchmark              # every configuration (about 2 h on a laptop CPU, mostly the CNN)
python -m experiments.benchmark ls9_calib    # one configuration
python -m experiments.summarize              # results table + paired Wilcoxon tests
python -m experiments.replay_check           # the app model on subject 05, as in the app
```

`experiments/benchmark.py` documents every configuration: task (7 classes,
"idle" = unmarked and rest merged, or the 6 labelled gestures), feature set,
calibration, class-imbalance handling, classifier (Random Forest, Extra Trees,
LDA, LightGBM, XGBoost, 1D CNN) and optional smoothing of the output. The
committed `experiments/results/*.json` are the outputs of these commands.

The thesis model was trained in a notebook that is not part of this
repository; the model file is attached to the
[thesis-2026 release](https://github.com/Purn0/emg-ai-arm/releases/tag/thesis-2026).
`python -m experiments.replay_check --model rf_emg_best.joblib` reproduces its
52.6 % on subject 05.

## Repository layout

```
emg_ai_arm/
  acquisition/replay_csv_8ch.py   replay a recorded session as windows
  camera/                         webcam thread, MediaPipe + Random Forest recognizer
  control/                        RobotCommand, gesture -> command map
  emulator/fake_emg_8ch.py        synthetic smoke-test signal
  features/emg_features.py        thesis, Hudgins, TD4 and TD9 feature sets
  features/extract_features_8ch.py  the thesis features, one window at a time
  emg_model.py                    load the EMG model, per-user calibration
  visualization/                  PyQt6 app, signal plot, inference tab, arm
experiments/
  download_data.py                UCI dataset
  data.py                         loading, windowing, subject folds
  benchmark.py, cnn.py            subject-wise evaluation
  summarize.py                    results table and significance tests
  replay_check.py                 score a model on a replayed session
  train_models.py                 the app's EMG and camera models
  results/                        outputs of the benchmark
```

## Changes since the thesis

- EMG model: TD9 features and per-user calibration instead of the thesis
  features; trained on all subjects except 05, so the replay stays a test on
  an unseen person; evaluation code and results added (`experiments/`).
- Camera: frames are mirrored before recognition, as they were when the
  landmark data was collected (the thesis app did not mirror them); the 0.6
  confidence gate is active (it was implemented but set to 0); one forest
  pass per frame instead of two.
- Removed: the Trainer tab and the serial source (both from an earlier
  two-channel prototype and not compatible with the 8-channel model), a
  "Load gesture model" button that did not load anything, and unused
  prototype and placeholder modules. They remain in the `thesis-2026` tag.

## Data

EMG: Krilova, N., Kastalskiy, I., Kazantsev, V., Makarov, V., & Lobov, S.
(2018). *EMG Data for Gestures* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5ZP5C (CC BY 4.0).

## License

MIT, see [LICENSE](LICENSE).
