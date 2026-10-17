import os
import numpy as np
import torch
import open_clip
from PIL import Image

FPS = 2.0      # частота выборки кадров
WIN = 6        # окно = 6 кадров = 3 с
# Офлайн: положите веса локально и укажите путь в CLIP_WEIGHTS
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_WEIGHTS = os.getenv("CLIP_WEIGHTS", "laion2b_s34b_b79k")


class Embedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m, _, pre = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_WEIGHTS)
        self.model, self.pre = m.eval().to(self.device), pre

    @torch.no_grad()
    def __call__(self, frames_bgr, batch=32):
        out = []
        for i in range(0, len(frames_bgr), batch):
            x = torch.stack([self.pre(Image.fromarray(f[:, :, ::-1]))
                             for f in frames_bgr[i:i + batch]]).to(self.device)
            e = self.model.encode_image(x)
            out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out)


def window_feats(emb):
    """emb: (N, D) -> признаки окон (M, 3D) и индексы центральных кадров."""
    feats, centers = [], []
    for s in range(max(1, len(emb) - WIN + 1)):
        w = emb[s:s + WIN]
        feats.append(np.concatenate([w.mean(0), w.std(0), w[-1] - w[0]]))
        centers.append(s + len(w) // 2)
    return np.array(feats, dtype=np.float32), np.array(centers)
