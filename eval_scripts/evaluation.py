#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación en test set limpio:
- Exact Match (EM)
- SacreBLEU
- ChrF
- GLEU
Separando casos con y sin error.
"""

import json
import evaluate
from nltk.translate.gleu_score import sentence_gleu

# ===== CONFIG =====
TEST_SET_PATH = "/workspace/test_recovered_clean_adjusted_v2.jsonl"   # referencias
PRED_PATH     = "/workspace/preds_salamandra_stage2_A_largetest.jsonl"  # predicciones del modelo
OUT_JSON      = "results_salamandra_stage2_A_largetest.json"          # salida con métricas

# ===== MÉTRICAS =====
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def exact_match(preds, refs):
    return sum(p.strip() == r.strip() for p, r in zip(preds, refs)) / len(refs)

def gleu_score(preds, refs):
    """Calcula GLEU promedio sobre oraciones."""
    scores = []
    for p, r in zip(preds, refs):
        scores.append(sentence_gleu([r.split()], p.split()))
    return sum(scores) / len(scores)

def eval_subset(data, preds_dict, subset_name="all"):
    ids = [d["id"] for d in data if d["id"] in preds_dict]
    if not ids:  # <- nada que evaluar
        print(f"\n--- {subset_name.upper()} ---")
        print("⚠️ No hay ejemplos en este subconjunto.")
        return {
            "ExactMatch": None,
            "SacreBLEU": None,
            "ChrF": None,
            "GLEU": None
        }

    preds = [preds_dict[i] for i in ids]
    refs  = [next(d["correct"] for d in data if d["id"] == i) for i in ids]

    em = exact_match(preds, refs)
    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
    chrf_score = chrf.compute(predictions=preds, references=[[r] for r in refs])
    gleu = gleu_score(preds, refs)

    print(f"\n--- {subset_name.upper()} ---")
    print(f"Exact Match: {em*100:.2f}%")
    print(f"SacreBLEU:  {bleu_score['score']:.2f}")
    print(f"ChrF:       {chrf_score['score']:.2f}")
    print(f"GLEU:       {gleu*100:.2f}")

    return {
        "ExactMatch": em,
        "SacreBLEU": bleu_score['score'],
        "ChrF": chrf_score['score'],
        "GLEU": gleu * 100
    }

# ===== CARGA =====
data = [json.loads(l) for l in open(TEST_SET_PATH, encoding="utf-8")]
preds_dict = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(PRED_PATH, encoding="utf-8")}

# ===== SUBSETS =====
all_data   = data
noerr_data = [d for d in data if d["categories"] == []]  # no error
err_data   = [d for d in data if d["categories"] != []]  # había error

# ===== EVALUACIÓN =====
results = {
    "All": eval_subset(all_data, preds_dict, "All"),
    "NoError": eval_subset(noerr_data, preds_dict, "No Error"),
    "WithError": eval_subset(err_data, preds_dict, "With Error")
}

# ===== GUARDAR =====
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Resultados guardados en {OUT_JSON}")
