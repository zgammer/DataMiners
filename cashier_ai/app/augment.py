import cv2
import numpy as np


def random_view(frames, seed):
    """Один и тот же случайный «ракурс» для всех кадров видео:
    перспектива + поворот + масштаб (+ иногда зеркало).
    Имитирует съёмку под другим углом."""
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    j = lambda s: rng.uniform(-s, s)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = 0.12
    dst = src + np.float32([[j(d * w), j(d * h)] for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    R = cv2.getRotationMatrix2D((w / 2, h / 2), j(15), rng.uniform(0.9, 1.15))
    R = np.vstack([R, [0, 0, 1]])
    H = R @ M
    flip = rng.random() < 0.5
    out = []
    for f in frames:
        g = cv2.warpPerspective(f, H, (w, h), borderMode=cv2.BORDER_REFLECT)
        out.append(g[:, ::-1].copy() if flip else g)
    return out
