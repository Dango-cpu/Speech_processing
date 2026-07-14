from __future__ import annotations

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-vi-en"


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA was selected but is unavailable. Falling back to CPU.")
        return "cpu"
    return device


@st.cache_resource(show_spinner="Loading Helsinki-NLP/opus-mt-vi-en...")
def load_translation_model(device: str):
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_ID)
    model.to(device)
    model.eval()
    return tokenizer, model


def translate_vi_to_en(text: str, device: str = "auto") -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    resolved_device = resolve_device(device)
    tokenizer, model = load_translation_model(resolved_device)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(resolved_device)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, num_beams=4)
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()
