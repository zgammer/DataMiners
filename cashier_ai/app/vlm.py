"""Необязательный локальный VLM (Qwen2-VL-2B-Instruct, Apache-2.0) для описаний.
Веса заранее скачать и указать VLM_PATH. Если не задан — модуль отключён."""
import os
from PIL import Image

_VLM = None


def enabled():
    return bool(os.getenv("VLM_PATH"))


def _load():
    global _VLM
    if _VLM is None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        path = os.environ["VLM_PATH"]
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            path, torch_dtype="auto", device_map="auto", local_files_only=True)
        proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        _VLM = (model, proc)
    return _VLM


def describe(frames_bgr, hint):
    """Короткое фактическое описание кадров. Без домыслов и без идентификации лиц."""
    model, proc = _load()
    imgs = [Image.fromarray(f[:, :, ::-1]).resize((640, 360)) for f in frames_bgr]
    prompt = (
        "Это кадры с камеры над кассой банка. Опиши по-русски в 1-2 предложениях, "
        "ТОЛЬКО то, что видно: кто где находится и что делает с деньгами/документами. "
        "Не гадай и не описывай внешность людей. "
        f"Контекст от детектора: {hint}.")
    msgs = [{"role": "user",
             "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=120, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
