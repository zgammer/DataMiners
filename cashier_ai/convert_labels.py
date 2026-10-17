import cv2
import pandas as pd
from datetime import time
from pathlib import Path

SRC = "data/open_labels.xlsx"                      # путь к вашему Excel
COL_VIDEO, COL_VIOL, COL_TIME = "название", "нарушение", "таймкод"

df = pd.read_excel(SRC)
print("Колонки в файле:", df.columns.tolist())
print(df.head(), "\n")

def to_sec(x):
    if pd.isna(x): return None
    if isinstance(x, time): return x.hour * 3600 + x.minute * 60 + x.second
    if hasattr(x, "total_seconds"): return x.total_seconds()
    s = str(x).strip().replace(",", ".")
    if ":" in s:
        sec = 0
        for p in s.split(":"): sec = sec * 60 + float(p)
        return sec
    return float(s)

def to_flag(x):
    return 0 if str(x).strip().lower() in ("", "nan", "0", "нет", "норма", "нет нарушения") else 1

rows = []
for _, r in df.iterrows():
    sec = to_sec(r[COL_TIME])
    viol = 1 if sec is not None else to_flag(r[COL_VIOL])
    name = Path(str(r[COL_VIDEO]).strip()).stem
    rows.append(dict(video=name, violation=viol, timecode_sec=sec))

out = pd.DataFrame(rows)

# Проверки
for _, r in out.iterrows():
    p = Path("data/videos") / f"{r.video}.mov"
    if not p.exists():
        print("НЕТ ФАЙЛА:", p); continue
    cap = cv2.VideoCapture(str(p))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    if r.violation and pd.isna(r.timecode_sec):
        print("НАРУШЕНИЕ БЕЗ ТАЙМКОДА:", r.video)
    elif pd.notna(r.timecode_sec) and r.timecode_sec > dur:
        print(f"ТАЙМКОД ДЛИННЕЕ ВИДЕО: {r.video} ({r.timecode_sec} > {dur:.0f} c)")

out.to_csv("data/labels.csv", index=False)
print(f"\nГотово: {len(out)} видео, с нарушением: {out.violation.sum()}")