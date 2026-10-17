import base64, time
from pathlib import Path
import cv2, joblib, numpy as np

from . import vlm
from .features import FPS, Embedder, window_feats
from .rules import CLUSTER_TO_TYPE, rule_for
from .video import sample_frames

GAP = 2          # окон: склейка соседних срабатываний в одно событие
MAX_EVENTS = 5
TIME_BUDGET = 240  # с; жюри ждёт максимум 300

_emb = None
_model = None


def _load():
    global _emb, _model
    if _emb is None:
        _emb = Embedder()
        _model = joblib.load(Path("artifacts/model.joblib"))
    return _emb, _model


def _jpeg_b64(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def analyze(path, job_id=""):
    t0 = time.time()
    emb_model, M = _load()
    frames, ts = sample_frames(path, FPS)
    emb = emb_model(frames)
    F, centers = window_feats(emb)
    p = M["clf"].predict_proba(F)[:, 1]

    idx = np.where(p >= M["thr"])[0]
    groups = []
    for i in idx:
        if groups and i - groups[-1][-1] <= GAP:
            groups[-1].append(i)
        else:
            groups.append([i])
    groups.sort(key=lambda g: -p[g].max())

    violations = []
    for g in groups[:MAX_EVENTS]:
        peak = g[int(np.argmax(p[g]))]
        ci = int(centers[peak])
        cl = int(M["km"].predict(F[peak:peak + 1, :M["D"]])[0])
        tkey = CLUSTER_TO_TYPE.get(cl, "unknown")
        rule, src = rule_for(tkey)

        fi = sorted({max(0, ci - 1), ci, min(len(frames) - 1, ci + 1)})
        fi = [ci] + [i for i in fi if i != ci]
        evid = [dict(timestamp_sec=round(float(ts[i]), 1), image_base64=_jpeg_b64(frames[i]))
                for i in fi[:3]]

        desc = (f"Модель обнаружила на {ts[ci]:.1f} с признаки ситуации: {rule['what']} "
                f"(уверенность {p[peak]:.2f}). Это подсказка для проверки человеком.")
        if vlm.enabled() and time.time() - t0 < TIME_BUDGET - 40:
            try:
                seen = vlm.describe([frames[i] for i in fi[:3]], rule["what"])
                desc = f"На кадрах видно: {seen} {desc}"
            except Exception as e:  # VLM не должен ронять весь анализ
                print("VLM error:", e)

        violations.append(dict(type=rule["title"], timecode_sec=round(float(ts[ci]), 1),
                               confidence=round(float(p[peak]), 3), description=desc,
                               rule=src, evidence_frames=evid))

    violations.sort(key=lambda v: v["timecode_sec"])
    return dict(job_id=job_id, status="done",
                verdict="violation" if violations else "no_violation",
                violations=violations)
