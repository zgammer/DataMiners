import cv2

MAX_W = 960


def _resize(f):
    h, w = f.shape[:2]
    if w > MAX_W:
        f = cv2.resize(f, (MAX_W, int(h * MAX_W / w)), interpolation=cv2.INTER_AREA)
    return f


def sample_frames(path, fps=2.0):
    """Возвращает (список BGR-кадров, список таймстемпов в секундах)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео")
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src / fps))
    frames, ts, i = [], [], 0
    while cap.grab():
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                frames.append(_resize(f))
                ts.append(i / src)
        i += 1
    cap.release()
    if not frames:
        raise ValueError("В видео нет кадров")
    return frames, ts


def read_frame_at(path, t_sec):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ok, f = cap.read()
    cap.release()
    return _resize(f) if ok else None
