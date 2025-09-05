#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferencia sobre test_recovered_clean.jsonl para modelos instruidos.
Genera preds_model.jsonl con:
{id: ..., pred: "frase corrixida"}
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json
from tqdm import tqdm

# === Paths ===
model_path = "/workspace/corrector_salamandra_stage2_A"   # ruta al modelo FT
test_path  = "/workspace/test_recovered_clean_adjusted_v2.jsonl"      # test set limpio
out_path   = "/workspace/preds_salamandra_stage2_A_largetest.jsonl"        # salida

# === Cargar modelo y tokenizer ===
print(f"🔄 Cargando modelo desde {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    dtype=torch.float16,
)

# === Inferencia ===
results = []
with open(test_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Inferencia"):
        ex = json.loads(line)

        # Conversación según template de entrenamiento
        messages = [
            {
                "role": "system",
                "content": """Es un modelo especializado en corrixir exclusivamente os erros gramaticais e ortográficos en galego.\n Non debes cambiar nin engadir ningunha palabra se xa está correcta.\n O texto de saída debe ser idéntico ao orixinal, salvo nas correccións estritamente necesarias.\n Responde sempre en galego.\n\nSaída:\n\n'## Corrección: <texto>'\n'## Explicación: <texto>'."""
            },
            {
                "role": "user",
                "content": f"""{ex['prompt']}\n\n{ex['incorrect']}"""
            }
        ]

        # Crear prompt con el chat_template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generar salida
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,   # determinista en test
            temperature=0.0,
            top_p=1.0,
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decoded)

        # === Extraer solo "### Corrección:" ===
        pred = decoded
        if "### Corrección:" in decoded:
            pred = decoded.split("### Corrección:")[1]
            if "### Explicación:" in pred:
                pred = pred.split("### Explicación:")[0]
            pred = pred.strip()

        results.append({"id": ex["id"], "pred": pred})

# === Guardar predicciones ===
with open(out_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Predicciones guardadas en {out_path} ({len(results)} ejemplos)")
