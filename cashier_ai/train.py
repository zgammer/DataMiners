"""Обучение детектора нарушений на размеченном датасете.

labels.csv: video,violation,timecode_sec   (violation = 0/1; timecode пуст для 0)
Запуск: python train.py --videos data/videos --labels data/labels.csv --aug 2 --k 5
"""
import argparse, zlib
from pathlib import Path
import cv2, joblib, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.augment import random_view
from app.features import FPS, Embedder, window_feats
from app.video import read_frame_at, sample_frames

TOL = 3.0       # окно вокруг размеченного таймкода считаем положительным
NEG_GAP = 7.0   # дальше этого расстояния — уверенный негатив; между — игнор


def build_cache(vdir, labels, cache, n_aug):
    cache.mkdir(parents=True, exist_ok=True)
    emb = None
    for _, r in labels.iterrows():
        p = vdir / (r.video if Path(r.video).suffix else f"{r.video}.mov")
        for a in range(n_aug + 1):
            out = cache / f"{p.stem}_a{a}.npz"
            if out.exists():
                continue
            emb = emb or Embedder()
            frames, ts = sample_frames(p, FPS)
            if a:
                frames = random_view(frames, zlib.crc32(f"{p.stem}{a}".encode()))
            np.savez(out, emb=emb(frames), ts=np.array(ts))
            print("cached", out.name)


def make_dataset(vdir, labels, cache, n_aug):
    S = []
    for _, r in labels.iterrows():
        p = vdir / (r.video if Path(r.video).suffix else f"{r.video}.mov")
        for a in range(n_aug + 1):
            d = np.load(cache / f"{p.stem}_a{a}.npz")
            F, c = window_feats(d["emb"])
            tc = d["ts"][c]
            if int(r.violation):
                dist = np.abs(tc - float(r.timecode_sec))
                lab = np.where(dist <= TOL, 1, np.where(dist >= NEG_GAP, 0, -1))
            else:
                lab = np.zeros(len(F), int)
            S.append(dict(video=p.stem, path=p, aug=a, F=F, tc=tc, lab=lab,
                          viol=int(r.violation), t_true=r.get("timecode_sec")))
    return S


def fit(S):
    X = np.concatenate([s["F"][s["lab"] >= 0] for s in S])
    y = np.concatenate([s["lab"][s["lab"] >= 0] for s in S])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000))
    return clf.fit(X, y)


def cross_val(S):
    vids = sorted({s["video"] for s in S})
    folds = GroupKFold(n_splits=min(5, len(vids)))
    vid_arr = np.array(vids)
    scores, errs = {}, []
    for tr, te in folds.split(vid_arr, groups=vid_arr):
        tr_v, te_v = set(vid_arr[tr]), set(vid_arr[te])
        clf = fit([s for s in S if s["video"] in tr_v])
        for s in S:
            if s["video"] in te_v and s["aug"] == 0:
                p = clf.predict_proba(s["F"])[:, 1]
                scores[s["video"]] = (p.max(), s["viol"])
                if s["viol"]:
                    errs.append(abs(s["tc"][p.argmax()] - float(s["t_true"])))
    ys = np.array([v[1] for v in scores.values()])
    ps = np.array([v[0] for v in scores.values()])
    best = (0, 0.5)
    for thr in np.linspace(0.05, 0.95, 91):
        pred = ps >= thr
        tp, fp, fn = (pred & (ys == 1)).sum(), (pred & (ys == 0)).sum(), (~pred & (ys == 1)).sum()
        f2 = 5 * tp / max(1, 5 * tp + 4 * fn + fp)   # F2: пропуск хуже ложной тревоги
        best = max(best, (f2, thr))
    print(f"CV: F2={best[0]:.3f} thr={best[1]:.2f}; "
          f"медианная ошибка таймкода={np.median(errs):.2f} c")
    return float(best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cache", default="artifacts/cache")
    ap.add_argument("--aug", type=int, default=2)
    ap.add_argument("--k", type=int, default=5, help="число кластеров типов")
    a = ap.parse_args()

    labels = pd.read_csv(a.labels)
    vdir, cache = Path(a.videos), Path(a.cache)
    build_cache(vdir, labels, cache, a.aug)
    S = make_dataset(vdir, labels, cache, a.aug)

    thr = cross_val(S)
    clf = fit(S)

    # Кластеры положительных окон -> кандидаты в «типы нарушений»
    D = S[0]["F"].shape[1] // 3
    pos = [(s, i) for s in S if s["aug"] == 0 for i in np.where(s["lab"] == 1)[0]]
    P = np.array([s["F"][i, :D] for s, i in pos])
    km = KMeans(n_clusters=min(a.k, len(P)), n_init=10, random_state=0).fit(P)
    for k in range(km.n_clusters):
        outdir = Path("artifacts/clusters") / f"cluster_{k}"
        outdir.mkdir(parents=True, exist_ok=True)
        idx = np.where(km.labels_ == k)[0]
        idx = idx[np.argsort(np.linalg.norm(P[idx] - km.cluster_centers_[k], axis=1))][:8]
        for j in idx:
            s, i = pos[j]
            f = read_frame_at(s["path"], float(s["tc"][i]))
            if f is not None:
                cv2.imwrite(str(outdir / f"{s['video']}_{s['tc'][i]:.1f}.jpg"), f)
        print(f"cluster {k}: {len(np.where(km.labels_ == k)[0])} окон")

    joblib.dump(dict(clf=clf, thr=thr, km=km, D=D), "artifacts/model.joblib")
    print("Готово. Посмотрите artifacts/clusters/* и заполните CLUSTER_TO_TYPE в app/rules.py")


if __name__ == "__main__":
    main()
