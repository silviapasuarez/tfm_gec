#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFM GEC Unified CLI
===================
Pipeline de generación y filtrado de datos sintéticos para corrección gramatical en gallego,
unificando los scripts utilizados durante el TFM en una única interfaz de línea de comandos.

Subcomandos:
- references   → variantes gramaticalmente correctas a partir de 'text'
- avalingua    → filtra oraciones correctas con Linguakit (AvaLingua)
- error-gen    → genera pares (correct/incorrect) con errores controlados
- jsonfix      → repara un archivo con objetos sueltos → JSON array válido
- post-edit    → recalcula 'changes' con anclas duales + 'index' compatible
- token-score  → surprisal de 'correct' con minicons (MaskedLM)
- pipeline     → ejecuta una secuencia de pasos definida en YAML
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, csv, gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# --- imports perezosos ---
def _lazy_import_transformers():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    return torch, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def _lazy_import_tqdm():
    from tqdm import tqdm
    return tqdm

# -------------------------- Utilidades IO ------------------------------------
def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        head = f.read(1); f.seek(0)
        if head == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("El JSON no es una lista de objetos.")
            return data
        return [json.loads(line) for line in f if line.strip()]

def save_json_array(rows: List[Dict[str, Any]], path: str) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

# ------------------- Extractores JSON (GPT-OSS/DeepHermes) --------------------
def extract_generated_json_harmony_final(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    marker = "<|start|>assistant<|channel|>final<|message|>"
    end_marker = "<|return|>"
    idx = text.find(marker)
    if idx != -1:
        segment = text[idx + len(marker):].split(end_marker, 1)[0].strip()
    else:
        fallback_marker = "<|start|>assistant"
        end_fallback = "<|end|>"
        j = text.find(fallback_marker)
        if j == -1:
            return None
        segment = text[j + len(fallback_marker):].split(end_fallback, 1)[0].strip()
    for i in range(len(segment)):
        try:
            obj, _ = decoder.raw_decode(segment[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None

def extract_generated_json_mistral(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    idx = text.find(marker)
    if idx == -1: return None
    after = text[idx + len(marker):].strip()
    for i in range(len(after)):
        try:
            obj, _ = decoder.raw_decode(after[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None

# -------------------------- Prompts ------------------------------------------
SUPPORT_GUIDE = ("Support Guide: Galician Grammar Phenomena and Error Patterns\n\n"
                    "The following linguistic notes explain key aspects of Galician grammar that you should be aware of when generating or correcting text. These are Galician-specific phenomena, and examples are given in Galician.\n\n"
                    "1. Clitic pronouns (pronomes átonos) and their placement:\n"
                    "- Galician uses unstressed object pronouns (me, te, se, o, a, nos, vos, lle, lles, etc.) that attach to verbs.\n"
                    "- Placement rules are conditional proclisis (before the verb) or enclisis (after the verb), depending on the syntactic environment.\n"
                    "- Enclisis occurs mainly in: infinitives (querelo), gerunds (facéndoo), affirmative imperatives (dámeo).\n"
                    "- Proclisis occurs with: negation (non me digas), subordinate clauses (dixo que me chamou), certain adverbs (xa me avisaron), interrogatives (que me dixo?).\n"
                    "Example errors:\n"
                    "Correct: Non me chamou.  Incorrect: Non chamoume.\n"
                    "Correct: Quere velo.  Incorrect: O quere ver.\n\n"
                    "2. Agreement (concordancia):\n"
                    "- Number agreement: Nouns, determiners, adjectives, and verbs must agree in singular/plural.\n"
                    "- Gender agreement: Adjectives and determiners agree in masculine/feminine with the noun.\n"
                    "- Verbal agreement: Verbs agree with the subject in person and number.\n"
                    "Example errors:\n"
                    "Correct: As nenas intelixentes xogaban no parque.  Incorrect: As nena intelixentes xogaban no parque.\n"
                    "Correct: O rapaz é alto.  Incorrect: O rapaz é alta.\n\n"
                    "3. Contractions (contraccións):\n"
                    "- In Galician, certain prepositions and articles merge into contracted forms.\n"
                    "- Common contractions: a+o=ao, a+a=á, de+o=do, de+os=dos, en+a=na, con+o=co, con+os=cos.\n"
                    "Example errors:\n"
                    "Correct: Vou ao mercado.  Incorrect: Vou a o mercado.\n"
                    "Correct: Falou cos veciños.  Incorrect: Falou con os veciños.\n\n"
                    "4. Tense, aspect, and mood (tempo, aspecto e modo):\n"
                    "- Galician verbs conjugate for tense, aspect, and mood.\n"
                    "- Wrong tense/mood usage can change meaning; apply such changes moderately.\n"
                    "- Subjunctive is used in certain subordinate clauses after triggers like querer que, é posible que.\n"
                    "Example errors:\n"
                    "Correct: Quero que veñas mañá.  Incorrect: Quero que vés mañá.\n"
                    "Correct: Onte fun á praia.  Incorrect: Onte vou á praia.\n\n"
                    "5. Minor orthographic changes (ortografía leve):\n"
                    "- Small spelling mistakes can involve: missing/misplaced accents, incorrect vowel in diphthongs/hiatuses, misuse of diaeresis, hyphenation errors.\n"
                    "Example errors:\n"
                    "Correct: Está frío.  Incorrect: Esta frío.\n"
                    "Correct: Lingua galega.  Incorrect: Lingoa galega.\n\n"
                    "General guidance for generating errors:\n"
                    "1. Keep the rest of the sentence unchanged except for the targeted error(s).\n"
                    "2. Introduce 1–2 errors per sentence in most cases; occasionally more for training diversity.\n"
                    "3. Avoid adding or removing large chunks of meaning.\n"
                    "4. Ensure errors are plausible for a Galician speaker — not random noise.\n"
                    "5. Do not introduce Spanish or Portuguese vocabulary unless the error being simulated is a known castelanismo in Galician."
                    "6. Introduce some punctuation errors if possible, such as missing commas or periods, but do not change the overall structure of the sentence.\n"
                    "7. Introduce occasional errors in word order or sentence structure to simulate more advanced learner mistakes.\n"
                    "-------------------------------------------------------------\n\n"
)

def build_prompt_references(reference: str) -> Tuple[List[Dict[str, str]], str]:
    sys_content = ("You are a Galician writer. Given a sentence in Galician, your task is to generate a new sentence "
                    "using the given sentence as a reference but you should use different words and meanings. "
                    "The sentence must be grammatically correct and semantically plausible. Make the sentences look like fragments of novels. "
                    "Do not use Portuguese or Spanish, only Galician.\n\n"
                    "Only return a JSON object with the following fields:\n"
                    "1. `reference_text`: the original input sentence\n"
                    "2. `new_text`: the new generated sentence\n"
                    "3. `confidence`: a float between 0 and 1 indicating the model's confidence in the grammatical correctness and semantic plausibility of the new sentence\n\n"
                    "Do not include any extra explanation or commentary."
                    "Here is an example of the required format:\n\n"
                    "{\n"
                    "  \"reference_text\": \"O ceo está cuberto de nubes.\",\n"
                    "  \"new_text\": \"As nubes esténdense por todo o horizonte.\",\n"
                    "  \"confidence\": 0.95\n"
                    "}\n\n"
    )
    user_content = (
                    f"Reference sentence:\n{reference}\n\n"
                    "Generate a new sentence with different meaning but using the given sentence as a reference for syntax and grammar, but very different. "
                    "Sentence must be grammatically and semantically plausible and correct. Avoid using Portuguese or Spanish, use only Galician."
    )
    return [{"role": "system", "content": sys_content},
            {"role": "user", "content": user_content}], reference

def build_prompt_error_gen(correct_sentence: str) -> List[Dict[str, str]]:
    sys_content = (
        "You are an expert in Galician grammar. Your only task is to introduce grammatical errors "
        "in given Galician sentences and output a structured JSON. Do not include thoughts, explanations, or extra text."
        "Generate corrupted sentences that contain grammatical errors as if a second language learner wrote them."
        + "\n\n" + SUPPORT_GUIDE +
        "-------------------------------------------------------------\n\n"
        "Your output must be a JSON object with the following fields:\n"
        "1. `correct`: the original sentence\n"
        "2. `incorrect`: the sentence with grammatical errors\n"
        "3. `categories`: a list of error types tags introduced\n"
        "4. `changes`: a list of objects describing ONLY the changes made, where each object contains:\n"
        "   - `original`: the word in the original sentence\n"
        "   - `incorrect`: the modified word in the incorrect sentence\n"
        "   - `index`: the 0-based index of the word in the sentence\n"
        "5. `confidence`: A score from 0 to 1 indicating the model's certainty about the generated errors. Be fair in your confidence judgement.\n\n"
        "Do not add unchanged words to the `changes` list.\n"
        "Format your output exactly like this:\n\n"
        "{\n"
        f'  "correct": "{correct_sentence}",\n'
        '  "incorrect": "[corrupted sentence]",\n'
        '  "categories": ["type1", "type2"],\n'
        '  "changes": [\n'
        '    {"original": "word1", "incorrect": "word2", "index": 3}\n'
        '  ],\n'
        '  "confidence": [confidence score]\n'
        "}"
    )
    user_content = (
                    "You will be given a correct sentence written in Galician. Your task is to rewrite the sentence "
                    "by introducing diverse grammatical errors as if a second language learner wrote them. Include a good variety of Galician errors. Do not explain or add any extra text. Do not change any word that has not been altered. "
                    "Do not use Portuguese or Spanish, only Galician.\n\n"
                    "Here is the correct sentence:\n"
                    f"{correct_sentence}\n\n"
                    "Please rewrite it with grammatical errors."
    )
    return [{"role": "system", "content": sys_content},
            {"role": "user", "content": user_content}]

# -------------------------- references ----------------------------------------
def cmd_references(args: argparse.Namespace) -> None:
    torch, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig = _lazy_import_transformers()
    tqdm = _lazy_import_tqdm()

    inp = load_json_or_jsonl(args.input)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "gpt-oss-20b":
        model_id = "openai/gpt-oss-20b" if args.variant == "full" else "unsloth/gpt-oss-20b-bnb-4bit"
        extractor = extract_generated_json_harmony_final
        quant = None
    elif args.model == "deephermes":
        model_id = "NousResearch/DeepHermes-3-Mistral-24B-Preview"
        extractor = extract_generated_json_mistral
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        raise ValueError("Modelo no soportado.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True,
        quantization_config=quant, low_cpu_mem_usage=True,
    )

    with out_path.open("w", encoding="utf-8") as out_f, torch.inference_mode():
        for entry in tqdm(inp, desc="Generating references", unit="sent"):
            ref = (entry.get(args.input_field) or "").strip()
            if not ref: continue
            messages, _ = build_prompt_references(ref)
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt", return_attention_mask=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen_ids = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, do_sample=True,
                repetition_penalty=args.repetition_penalty, top_p=args.top_p if args.top_p else None,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True,
            )
            out_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
            parsed = extractor(out_text)
            if parsed and {"reference_text", "new_text"} <= set(parsed.keys()):
                rec = {"reference_text": parsed["reference_text"], "new_text": parsed["new_text"]}
                if "confidence" in parsed: rec["confidence"] = parsed["confidence"]
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); out_f.flush()

            del inputs, gen_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            gc.collect()

# -------------------------- avalingua -----------------------------------------
def run_avalingua_sentence(text: str, linguakit_bin: str, timeout: int = 10) -> Dict[str, Any]:
    import subprocess
    res = subprocess.run(
        [linguakit_bin, "aval", "gl", "-s", "-json"],
        input=text, text=True, capture_output=True, check=True, timeout=timeout
    )
    return json.loads(res.stdout)

def cmd_avalingua(args: argparse.Namespace) -> None:
    tqdm = _lazy_import_tqdm()
    data = load_json_or_jsonl(args.input)
    kept, errors = [], []
    for idx, item in enumerate(tqdm(data, desc="AvaLingua check", unit="sent")):
        text = (item.get(args.field) or "").strip()
        if not text:
            errors.append({"index": idx, "error": "empty"}); continue
        try:
            out = run_avalingua_sentence(text, args.linguakit_bin, timeout=args.timeout)
            stats = out.get("avalingua", {}).get("statistics", {})
            total = (stats.get("grammar_errors", 0) or 0) + (stats.get("lexical_errors", 0) or 0)
            if total == 0: kept.append(item)
        except Exception as e:
            errors.append({"index": idx, "error": repr(e)})
    save_json_array(kept, args.output)
    if args.errlog: save_json_array(errors, args.errlog)
    print(f"Kept {len(kept)} items; logged {len(errors)} errors.")

# -------------------------- error-gen -----------------------------------------
def cmd_error_gen(args: argparse.Namespace) -> None:
    torch, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig = _lazy_import_transformers()
    tqdm = _lazy_import_tqdm()

    inp = load_json_or_jsonl(args.input)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "gpt-oss-20b":
        model_id = "openai/gpt-oss-20b" if args.variant == "full" else "unsloth/gpt-oss-20b-bnb-4bit"
        extractor = extract_generated_json_harmony_final; quant = None
    elif args.model == "deephermes":
        model_id = "NousResearch/DeepHermes-3-Mistral-24B-Preview"
        extractor = extract_generated_json_mistral
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        raise ValueError("Modelo no soportado.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True, quantization_config=quant, low_cpu_mem_usage=True
    )

    with out_path.open("w", encoding="utf-8") as out_f, torch.inference_mode():
        for entry in tqdm(inp, desc="Generating errors", unit="sent"):
            correct = (entry.get(args.field_for_correct) or "").strip()
            if not correct: continue
            messages = build_prompt_error_gen(correct)
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt", return_attention_mask=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen_ids = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                repetition_penalty=args.repetition_penalty, do_sample=True,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id
            )
            out_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
            parsed = extractor(out_text)
            if parsed and all(k in parsed for k in ("correct", "incorrect", "categories", "changes", "confidence")):
                out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n"); out_f.flush()
            del inputs, gen_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            gc.collect()

# -------------------------- jsonfix (micro-fixes) ------------------------------
SMART_QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": '"', "\u2019": '"', "\u00AB": '"', "\u00BB": '"'}
_UNQUOTED_KEY_RE = re.compile(r'([{\s,])([A-Za-z_][A-Za-z0-9_\-]*)(\s*):')

def normalize_quotes(s: str) -> str:
    for b,g in SMART_QUOTES.items(): s = s.replace(b,g)
    return s
def strip_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[\}\]])", r"\1", s)
def quote_unquoted_keys(s: str) -> str:
    def repl(m): return f'{m.group(1)}"{m.group(2)}"{m.group(3)}:'
    prev=None
    while prev!=s: prev,s=s,_UNQUOTED_KEY_RE.sub(repl,s)
    return s
def micro_fix(text: str) -> str:
    return quote_unquoted_keys(strip_trailing_commas(normalize_quotes(text)))
def extract_json_objects(raw: str) -> List[str]:
    out=[];depth=0;in_str=False;esc=False;start=None
    for i,ch in enumerate(raw):
        if in_str:
            if esc: esc=False
            elif ch=="\\": esc=True
            elif ch=='"': in_str=False
            continue
        else:
            if ch=='"': in_str=True; continue
            if ch=='{':
                if depth==0: start=i
                depth+=1
            elif ch=='}' and depth>0:
                depth-=1
                if depth==0 and start is not None:
                    out.append(raw[start:i+1]); start=None
    return out
def cmd_jsonfix(args: argparse.Namespace) -> None:
    raw = Path(args.input).read_text(encoding="utf-8", errors="replace")
    if raw.rstrip().endswith("{"): raw = raw.rstrip()[:-1]
    blobs = extract_json_objects(raw); valid=[]; discarded=0
    disc = Path(args.output).with_suffix(Path(args.output).suffix + ".discarded.jsonl")
    for blob in blobs:
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict): valid.append(obj); continue
        except Exception: pass
        fixed = micro_fix(blob)
        try:
            obj = json.loads(fixed)
            if isinstance(obj, dict): valid.append(obj); continue
        except Exception as e:
            with disc.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"__error__": str(e), "__snippet__": fixed[:2000]}, ensure_ascii=False)+"\n")
            discarded += 1
    save_json_array(valid, args.output)
    if discarded: print(f"Descartados: {discarded} → {disc}")
    print(f"Objetos válidos: {len(valid)}")

# -------------------------- post-edit (changes) -------------------------------
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
PUNCT_RIGHT_GLUE = set(",.;:!?…%)]}»”’"); PUNCT_LEFT_GLUE = set("([{«“‘#")
def tokenize(text: str) -> List[str]: return TOKEN_PATTERN.findall(text or "")
def detok(tokens: List[str]) -> str:
    out=[]; 
    for i,tok in enumerate(tokens):
        if i==0: out.append(tok); continue
        prev = out[-1]
        if tok in PUNCT_RIGHT_GLUE: out[-1]=prev+tok
        elif prev and prev[-1] in PUNCT_LEFT_GLUE: out[-1]=prev+tok
        else: out.append(" "+tok)
    return "".join(out)
def _emit_change(changes: List[Dict], c_span: List[str], i_span: List[str], i1:int, j1:int, tag:str, index_source:Optional[str]=None)->None:
    e={"original":detok(c_span) if c_span else "","incorrect":detok(i_span) if i_span else "",
       "index_correct":i1,"index_incorrect":j1,"len_correct":len(c_span),"len_incorrect":len(i_span),"op":tag}
    if index_source in {"incorrect","correct"}: e["index"]= j1 if index_source=="incorrect" else i1
    changes.append(e)
def compute_changes(correct:str, incorrect:str, index_source:Optional[str]=None, split_replacements:bool=True)->List[Dict]:
    from difflib import SequenceMatcher
    c_toks=tokenize(correct); i_toks=tokenize(incorrect)
    sm=SequenceMatcher(a=c_toks,b=i_toks,autojunk=False); changes=[]
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=="equal": continue
        c_span=c_toks[i1:i2]; i_span=i_toks[j1:j2]
        if tag=="replace" and split_replacements:
            if len(c_span)==len(i_span):
                for k in range(len(c_span)):
                    if c_span[k]==i_span[k]: continue
                    _emit_change(changes,[c_span[k]],[i_span[k]],i1+k,j1+k,"replace",index_source)
                continue
            inner=SequenceMatcher(a=c_span,b=i_span,autojunk=False); any_split=False
            for itag,ii1,ii2,jj1,jj2 in inner.get_opcodes():
                if itag=="equal": continue
                any_split=True; _emit_change(changes,c_span[ii1:ii2],i_span[jj1:jj2],i1+ii1,j1+jj1,itag,index_source)
            if any_split: continue
        _emit_change(changes,c_span,i_span,i1,j1,tag,index_source)
    return changes
def fix_record(obj: Dict, index_source: Optional[str], split_replacements: bool) -> Dict:
    obj["changes"] = compute_changes(obj.get("correct",""), obj.get("incorrect",""),
                                     index_source=index_source, split_replacements=split_replacements)
    if "categories" in obj and not isinstance(obj["categories"], list):
        obj["categories"]=[obj["categories"]]
    return obj
def cmd_post_edit(args: argparse.Namespace) -> None:
    raw = Path(args.input).read_text(encoding="utf-8")
    try: parsed = json.loads(raw)
    except json.JSONDecodeError: parsed=None
    split_repl = not args.no_split_replacements
    if isinstance(parsed, list):
        fixed=[fix_record(dict(it), args.compat_index, split_repl) for it in parsed]
        save_json_array(fixed, args.output); print(f"Procesados {len(fixed)} items (JSON array).")
    elif isinstance(parsed, dict):
        fixed=fix_record(parsed, args.compat_index, split_repl)
        save_json_array([fixed], args.output); print("Procesado 1 objeto (JSON).")
    else:
        out_lines=[]
        for line in raw.splitlines():
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line); fixed=fix_record(obj, args.compat_index, split_repl)
                out_lines.append(json.dumps(fixed, ensure_ascii=False))
            except json.JSONDecodeError:
                out_lines.append(line)
        Path(args.output).write_text("\n".join(out_lines)+("\n" if out_lines else ""), encoding="utf-8")
        print(f"Procesadas {len(out_lines)} líneas (JSONL).")

# -------------------------- token-score (minicons) ----------------------------
def cmd_token_score(args: argparse.Namespace) -> None:
    from minicons import scorer
    import numpy as np
    tqdm = _lazy_import_tqdm()
    inp = Path(args.input); rows=[]
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: continue
    texts, idx_map = [], []
    for i, ex in enumerate(rows):
        c=(ex.get("correct") or "").strip()
        if c: texts.append(c); idx_map.append(i)
    print(f"Scoring {len(texts)} sentences with {args.model_id} in batches of {args.batch_size}...")
    mlm_scorer = scorer.MaskedLMScorer(args.model_id, device=args.device)
    def batched_surprisal(mlm_scorer, texts, batch_size=64):
        all_scores=[]
        for i in tqdm(range(0,len(texts),batch_size), desc='Scoring (batched)'):
            batch=texts[i:i+batch_size]
            scores=mlm_scorer.sequence_score(batch, reduction=lambda x: float(np.mean([t.item() for t in x])))
            all_scores.extend([-s for s in scores])
        return all_scores
    surprisals = batched_surprisal(mlm_scorer, texts, batch_size=args.batch_size)
    outd=Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
    csv_path=outd/(inp.stem+"_scores.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["id","correct","surprisal"])
        for local_i, global_i in enumerate(idx_map):
            ex=rows[global_i]; w.writerow([ex.get("id", global_i+1), ex.get("correct",""), surprisals[local_i]])
    print(f"CSV guardado en: {csv_path}")

# -------------------------- pipeline (YAML) -----------------------------------
def cmd_pipeline(args: argparse.Namespace) -> None:
    try:
        import yaml
    except Exception as e:
        raise SystemExit("pyyaml no está instalado: pip install pyyaml") from e
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for step in cfg.get("steps", []):
        name=step.get("name",""); kind=step.get("kind",""); params=step.get("params",{}) or {}
        print(f"\n=== Paso: {name} [{kind}] ===")
        if kind=="references": cmd_references(argparse.Namespace(**params))
        elif kind=="avalingua": cmd_avalingua(argparse.Namespace(**params))
        elif kind=="error-gen": cmd_error_gen(argparse.Namespace(**params))
        elif kind=="jsonfix": cmd_jsonfix(argparse.Namespace(**params))
        elif kind=="post-edit": cmd_post_edit(argparse.Namespace(**params))
        elif kind=="token-score": cmd_token_score(argparse.Namespace(**params))
        else: raise ValueError(f"Tipo de paso no soportado: {kind}")
    print("\n✅ Pipeline completado.")

# -------------------------- CLI ----------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tfm-gec", description="Pipeline unificada para GEC en gallego (TFM).")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("references", help="Generar variantes de referencia (correctas).")
    p.add_argument("--model", choices=["gpt-oss-20b","deephermes"], required=True)
    p.add_argument("--variant", choices=["bnb-4bit","full"], default="bnb-4bit")
    p.add_argument("--input", required=True)
    p.add_argument("--input-field", default="text")
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--offload", default=None)
    p.add_argument("--max-mem", default=None)
    p.add_argument("--cpu-mem", default="128GiB")
    p.add_argument("--keep-invalid", action="store_true")
    p.set_defaults(func=cmd_references)

    p = sub.add_parser("avalingua", help="Filtrar oraciones correctas con Linguakit (AvaLingua).")
    p.add_argument("--input", required=True)
    p.add_argument("--field", default="new_text")
    p.add_argument("--linguakit-bin", required=True)
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--output", required=True)
    p.add_argument("--errlog", default=None)
    p.set_defaults(func=cmd_avalingua)

    p = sub.add_parser("error-gen", help="Generar pares (correct/incorrect) con errores introducidos.")
    p.add_argument("--model", choices=["gpt-oss-20b","deephermes"], required=True)
    p.add_argument("--variant", choices=["bnb-4bit","full"], default="bnb-4bit")
    p.add_argument("--input", required=True)
    p.add_argument("--field-for-correct", default="new_text")
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--offload", default=None)
    p.add_argument("--keep-invalid", action="store_true")
    p.set_defaults(func=cmd_error_gen)

    p = sub.add_parser("jsonfix", help="Arreglar archivo con objetos sueltos → JSON array.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_jsonfix)

    p = sub.add_parser("post-edit", help="Recalcular 'changes' y anclas index_* a partir de correct/incorrect.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--compat-index", choices=["incorrect","correct"], default=None)
    p.add_argument("--no-split-replacements", action="store_true")
    p.set_defaults(func=cmd_post_edit)

    p = sub.add_parser("token-score", help="Calcular surprisal en 'correct' con minicons (MLM).")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-id", default="marcosgg/bert-base-gl-cased")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=cmd_token_score)

    p = sub.add_parser("pipeline", help="Ejecutar pipeline multi-paso desde YAML.")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_pipeline)

    return ap

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
