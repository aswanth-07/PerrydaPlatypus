# `src/`: the scored `predict()`

Everything the evaluator runs is in this folder: `main.py` (the `predict()` function), the three modules it uses
(`holistic_kp.py` keypoints, `normalise.py` normalisation, `extract.py` video reading) and `models/`:
`echogesture_int8.onnx` (the EchoGesture model, 2,000 ASL words, 3.32 MB INT8), `labels.json`, `sets.json` (the
same-sign set of each word), `export.json` (how it was exported and what it scored) and
`holistic_landmarker.task` (MediaPipe Holistic). Nothing is downloaded at run time and nothing leaves
the machine.

## Run it (Python 3.11)

```bash
python -m venv .venv
.venv/bin/pip install -r src/requirements.txt          # Windows: .venv\Scripts\pip
python src/main.py path/to/clips/ --out results/predictions.jsonl
```

`path` is one `.mp4` or a folder of them. Each clip becomes one JSON line, as the brief specifies:

```json
{"clip_id": "68400", "prediction": "READ", "top5": ["READ", "BUTTER", "RESEARCH", "PAINT", "FORK"],
 "confidence": 0.9662, "latency_ms": 8.61, "pipeline_ms": 812.4, "keypoints_ms_per_frame": 32.1}
```

From Python:

```python
import sys; sys.path.insert(0, "src")
from main import predict
predict("clip.mp4")
```

- `clip_id` is the file name without extension; `prediction` and `top5` are uppercase words from the model's
  fixed vocabulary (`models/labels.json`, the 2,000 WLASL words).
- `latency_ms` is the INT8 model's inference time for the clip (the figure the brief's 50 ms budget is about).
  `pipeline_ms` is the whole clip on one CPU thread: decoding, MediaPipe Holistic on each landmarked frame and the
  model. `keypoints_ms_per_frame` is the Holistic cost per frame, which is what a live camera pays per frame.

## Options (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `ECHO_VOCAB_FILE` | unset | a word list (one per line or a JSON list): answers are restricted to those words |
| `ECHO_FRAMES` | `96` | frames landmarked per clip: every frame up to 96, then evenly spaced. `24` is 2.8x faster per clip and costs 1.2 points top-1 (63.45% vs 64.63% on the 2,878 WLASL2000 test videos) |
| `ECHO_SET_RULE` | `1` | `0` answers from the network's own probabilities instead of the same-sign answer rule (see the main README) |
| `ECHO_FRESH_LANDMARKER` | unset | `1` builds a new Holistic landmarker per clip (same accuracy, slower) |
| `ECHO_LANES` | `1` | `2`: two Holistic landmarkers take the clip's frames in turn, in parallel threads; 1.8x faster per clip, 78.8% vs 79.5% top-1 on the first 600 test clips (within noise). Published numbers use 1 |

## Docker

From the repository root: `docker build -t echogesture .` then
`docker run --rm -v "$PWD/clips:/clips" -v "$PWD/results:/out" echogesture /clips --out /out/predictions.jsonl`.

Training, export, evaluation and the Android app are described in the repository's top-level `README.md`.
