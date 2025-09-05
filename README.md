# tfm_gec
Repositorio del TFM "Primeros pasos hacia la corrección gramatical y ortográfica en gallego con LLM mediante la creación de conjuntos de datos sintéticos"

## 1. Características

- **Generación de variantes de referencia** (sentencias correctas, estilo y tema independiente del original).
- **Filtrado gramatical** con **AvaLingua** (Linguakit) manteniendo la estructura de los datos.
- **Inserción controlada de errores** (pares `correct`/`incorrect` con `categories` y `changes`).
- **Reparación de JSON** heterogéneos → **JSON array** válido.
- **Post-edición de índices**: recálculo robusto de `changes` con anclas duales `index_correct/index_incorrect` y opción de índice compatible (`--compat-index`).
- **Puntuación de surprisal** token‑basado (MLM, `minicons`) sobre el lado *correct*.

---

## 2. Requisitos

- Python ≥ 3.10
- Paquetes:
  ```bash
  pip install torch transformers accelerate bitsandbytes tqdm pyyaml minicons
  ```
- **Modelos HF** (descarga automática):
  - `NousResearch/DeepHermes-3-Mistral-24B-Preview` (por defecto en ejemplos)
  - `openai/gpt-oss-20b` o `unsloth/gpt-oss-20b-bnb-4bit` (opcional)
- **Linguakit** instalado y accesible para `avalingua` (ej.: `/home/usuario/Linguakit/linguakit`).

> GPU recomendada: ≥ 24 GB VRAM para 4‑bit; para cargas mayores, ajustar `--offload`/`--max-mem` y cuantización 4‑bit.

---

## 3. Estructura del repositorio

```
.
├── tfm_gec_cli.py         # CLI unificada (entry script)
└── README.md              # Este documento
```

---

## 4. Formatos de entrada/salida

### 4.1. Entrada para `references`
- **JSONL/JSON** con un campo de texto de entrada (por defecto, `text`).  
  Ej. (`orixinais.jsonl`):
  ```json
  {"id": 1, "text": "O vento amaina na ribeira do Tambre."}
  {"id": 2, "text": "As praias baleiras gardan ecos do inverno."}
  ```

### 4.2. Salida de `references`
- **JSONL** con campos mínimos:
  ```json
  {"reference_text": "O vento amaina na ribeira do Tambre.", "new_text": "O río baixa manso e a brétema vai cedendo.", "confidence": 0.86}
  ```

### 4.3. Filtrado `avalingua`
- **Entrada**: JSONL/JSON con un campo (`--field`) a verificar, p. ej. `new_text`.
- **Salida**: **JSON array** (conserva el resto de campos originales) solo con oraciones sin errores.

### 4.4. Generación de pares `error-gen`
- **Entrada**: el resultado filtrado de `avalingua` con campo correcto (por defecto `new_text`).
- **Salida**: **JSONL** con objetos del tipo:
  ```json
  {
    "correct": "A lúa ergueuse por riba dos montes.",
    "incorrect": "A lúa erguer por riba do montes.",
    "categories": ["agreement", "verbal_errors", "contractions"],
    "changes": [
      {"original": "ergueuse", "incorrect": "erguer", "index": 2},
      {"original": "dos", "incorrect": "do", "index": 5}
    ],
    "confidence": 0.71
  }
  ```

### 4.5. Post-edición `post-edit`
- **Entrada**: JSON/JSONL con `correct` y `incorrect`.
- **Salida**: mismo objeto añadiendo/reconstruyendo `changes` por *diff* de tokens, con:
  - `index_correct`, `index_incorrect`, `len_correct`, `len_incorrect`, `op ∈ {insert, delete, replace}`
  - Opción `--compat-index {incorrect,correct}` para emitir también `index` único.

### 4.6. Puntuación `token-score`
- **Entrada**: JSONL con campo `correct`.
- **Salida**: CSV (`*_scores.csv`) con columnas `id, correct, surprisal` (media negativa de log‑prob).

---

## 5. Uso rápido (paso a paso)

```bash
# 0) Entorno
python -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate bitsandbytes tqdm pyyaml minicons

# 1) Variantes de referencia (sentencias correctas)
python tfm_gec_cli.py references \
  --model deephermes \
  --input data/orixinais.jsonl --input-field text \
  --output outputs/references.jsonl

# 2) Filtrado gramatical con AvaLingua (Linguakit)
python tfm_gec_cli.py avalingua \
  --input outputs/references.jsonl --field new_text \
  --linguakit-bin /ruta/a/Linguakit/linguakit \
  --output outputs/references_avalingua.json

# 3) Inserción de errores (pares correct/incorrect)
python tfm_gec_cli.py error-gen \
  --model deephermes \
  --input outputs/references_avalingua.json --field-for-correct new_text \
  --output outputs/pairs_with_errors.jsonl

# 4) Post-edición de índices/cambios
python tfm_gec_cli.py post-edit \
  --input outputs/pairs_with_errors.jsonl \
  --output outputs/pairs_with_errors_post.jsonl \
  --compat-index incorrect

# 5) (Opcional) Surprisal de 'correct'
python tfm_gec_cli.py token-score \
  --input outputs/pairs_with_errors_post.jsonl \
  --out-dir outputs/scores \
  --model-id marcosgg/bert-base-gl-cased \
  --device cuda --batch-size 64
```
---
