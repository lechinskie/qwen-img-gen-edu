"""
Pipeline: PorSimplesSent -> similaridade e escolha da versão (3 níveis) -> Qwen-Image 2.1
          -> legenda + triagem infantil (Qwen3-VL) -> BERTScore -> relatório HTML.

Requisitos: pip install -r requirements.txt   (diffusers>=0.41 traz o QwenImage21Pipeline)
Uso: python main.py --n 5       (a base é baixada para dados/ na primeira execução)
A geração (prompts e imagens) e a legenda gravam cache em saida/; rode de novo para continuar de onde parou.

Memória (RTX 3050 6 GB): a codificação do prompt roda na CPU (Qwen3-VL 8B em BF16, ~18 GB de RAM);
a GPU só é usada na geração (transformer em NF4, ~um quarto dos 14 GB em BF16) e na legenda (Qwen3-VL 4B em NF4).
"""
import argparse
import csv
import gc
import html
import json
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

IMG_MODEL = "Qwen/Qwen-Image-2.1"
VL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
ST_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BERT_MODEL = "neuralmind/bert-base-portuguese-cased"  # BERTimbau base, 12 camadas
TSV_URL = "https://raw.githubusercontent.com/sidleal/porsimplessent/HEAD/pss/triplets_length.tsv"
LEVELS = {"original": "original_text", "natural": "natural_text", "strong": "strong_text"}
# Prompt estruturado: só o campo {frase} muda entre os níveis. As restrições vão no prompt positivo
# porque o Qwen-Image 2.1 roda sem CFG (true_cfg_scale=1), e então o negative_prompt é ignorado.
PROMPT = (
    "Ilustre a frase a seguir em uma única cena.\n"
    "Frase: {frase}\n"
    "Estilo: ilustração infantil de livro didático, traços simples, cores suaves.\n"
    "Composição: personagens e objetos principais em destaque, fundo simples.\n"
    "Restrições: sem texto escrito, sem violência ou elementos assustadores, "
    "adequado para crianças de 6 a 10 anos."
)
Q_CAPTION = "Descreva em uma frase curta, em português, o que aparece nesta imagem."
Q_KIDS = ("Esta imagem é adequada para crianças de 6 a 10 anos, sem violência, medo ou conteúdo impróprio? "
          "Responda apenas SIM ou NAO.")
SEED = 42
OUT = Path("saida")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def nf4(component):
    from diffusers import PipelineQuantizationConfig
    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=[component],
    )


@lru_cache(maxsize=1)
def sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ST_MODEL, device="cpu")


# ---------- Etapa 1: base -> amostra ----------
def load_base(tsv):
    tsv = Path(tsv)
    if not tsv.exists():
        tsv.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TSV_URL, tsv)
    return pd.read_csv(tsv, sep="\t", quoting=csv.QUOTE_NONE, encoding="utf-8")


def eda(df):
    """Reproduz os números da análise exploratória (seminário)."""
    both = df[(df.changed_ori_nat == "S") & (df.changed_nat_str == "S")]
    none = df[(df.changed_ori_nat == "N") & (df.changed_nat_str == "N")]
    words = lambda col: both[col].str.split().str.len().mean()
    print(f"{len(df)} triplas | {len(both)} simplificadas nos 2 níveis | {len(none)} sem alteração | "
          f"{words('original_text'):.1f} -> {words('strong_text'):.1f} palavras/frase (original -> forte)")


def load_items(df, n, seed=SEED, ids=None):
    """Triplas simplificadas nos dois níveis, com frases de tamanho médio (sorteadas, ou as de `ids`)."""
    df = df[(df.changed_ori_nat == "S") & (df.changed_nat_str == "S")]
    df = df[df.original_text.str.split().str.len().between(8, 30)]
    df = df.loc[[int(i.lstrip("s")) for i in ids]] if ids else df.sample(n=n, random_state=seed)
    return [
        {"id": f"s{idx:05d}", **{lvl: row[col] for lvl, col in LEVELS.items()}}
        for idx, row in df.iterrows()
    ]


# ---------- Etapa 2: similaridade (cosseno) e escolha da versão ----------
def stage_similarity(items, tau):
    """Cosseno de cada nível contra o original; escolhe a simplificação mais forte com cos >= tau."""
    st = sbert()
    for it in items:
        vec = {lvl: st.encode(it[lvl], convert_to_tensor=True) for lvl in LEVELS}
        it["cos"] = {lvl: round(float(st.similarity(vec[lvl], vec["original"])), 4) for lvl in LEVELS}
        it["escolhida"] = next((lvl for lvl in ("strong", "natural") if it["cos"][lvl] >= tau), "original")


# ---------- Etapa 3a: texto -> embeddings do próprio Qwen-Image (na CPU) ----------
def stage_prompts(items):
    path = OUT / "prompt_embeds.pt"
    cache = torch.load(path) if path.exists() else {}
    todo = [(it["id"], lvl, it[lvl]) for it in items for lvl in LEVELS if f'{it["id"]}|{lvl}' not in cache]
    if not todo:
        return
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(
        IMG_MODEL, transformer=None, vae=None, dtype=torch.bfloat16
    )
    with torch.no_grad():
        for sid, lvl, text in todo:
            emb, mask, _ = pipe.encode_prompt(PROMPT.format(frase=text), device="cpu")
            cache[f"{sid}|{lvl}"] = (emb.cpu(), None if mask is None else mask.cpu())
    torch.save(cache, path)
    del pipe
    free()


# ---------- Etapa 3b: embeddings -> imagem (transformer em NF4) ----------
def stage_images(items, size, steps):
    todo = [(it["id"], lvl) for it in items for lvl in LEVELS
            if not (OUT / "img" / f"{it['id']}_{lvl}.png").exists()]
    if not todo:
        return
    from diffusers import QwenImage21Pipeline
    cache = torch.load(OUT / "prompt_embeds.pt")
    pipe = QwenImage21Pipeline.from_pretrained(
        IMG_MODEL, text_encoder=None, dtype=torch.bfloat16,
        quantization_config=nf4("transformer"),
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    for sid, lvl in todo:
        emb, mask = cache[f"{sid}|{lvl}"]
        image = pipe(
            prompt_embeds=emb.to("cuda"),
            prompt_embeds_mask=None if mask is None else mask.to("cuda"),
            width=size, height=size, num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(SEED),  # mesma seed nos 3 níveis
        ).images[0]
        image.save(OUT / "img" / f"{sid}_{lvl}.png")
    del pipe
    free()


# ---------- Etapa 4a: imagem -> legenda em português + triagem infantil (Qwen3-VL) ----------
def ask_vl(model, proc, img, question, max_new_tokens):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": question}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def stage_captions(items):
    path = OUT / "captions.json"
    caps = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [(it["id"], lvl) for it in items for lvl in LEVELS if f'{it["id"]}|{lvl}' not in caps]
    if not todo:
        return caps
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        VL_MODEL, device_map={"": 0},  # tudo na GPU; falha cedo se faltar VRAM em vez de espalhar na CPU
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16),
    )
    proc = AutoProcessor.from_pretrained(VL_MODEL)
    for sid, lvl in todo:
        img = Image.open(OUT / "img" / f"{sid}_{lvl}.png").convert("RGB")
        kids = ask_vl(model, proc, img, Q_KIDS, 5)
        caps[f"{sid}|{lvl}"] = {
            "legenda": ask_vl(model, proc, img, Q_CAPTION, 80),
            "infantil": "SIM" if kids.upper().startswith("SIM") else "NAO",
        }
    path.write_text(json.dumps(caps, ensure_ascii=False, indent=1), encoding="utf-8")
    del model
    free()
    return caps


# ---------- Etapa 4b: verificação semântica (legenda x frase) ----------
def stage_metrics(items, caps):
    from bert_score import BERTScorer
    scorer = BERTScorer(model_type=BERT_MODEL, num_layers=12, lang="pt", device="cpu")
    st = sbert()
    keys = [(it, lvl) for it in items for lvl in LEVELS]
    cap = [caps[f'{it["id"]}|{lvl}']["legenda"] for it, lvl in keys]
    frase = [it[lvl] for it, lvl in keys]
    orig = [it["original"] for it, _ in keys]
    f_lvl = scorer.score(cap, frase)[2].tolist()
    f_ori = scorer.score(cap, orig)[2].tolist()
    cos_cap = st.similarity(st.encode(cap, convert_to_tensor=True), st.encode(orig, convert_to_tensor=True)).diagonal()
    df = pd.DataFrame([{
        "id": it["id"], "nivel": lvl, "escolhida": it["escolhida"] == lvl,
        "frase": it[lvl], "legenda": cap[i], "infantil": caps[f'{it["id"]}|{lvl}']["infantil"],
        "n_palavras": len(it[lvl].split()),
        "cos_texto_vs_original": it["cos"][lvl],
        "cos_legenda_vs_original": round(float(cos_cap[i]), 4),
        "bertscore_f1_vs_frase": f_lvl[i],
        "bertscore_f1_vs_original": f_ori[i],
    } for i, (it, lvl) in enumerate(keys)])
    df.to_csv(OUT / "resultados.csv", index=False, encoding="utf-8-sig")
    cols = ["n_palavras", "cos_texto_vs_original", "cos_legenda_vs_original",
            "bertscore_f1_vs_frase", "bertscore_f1_vs_original"]
    tab = df.groupby("nivel")[cols].mean()
    tab.loc["escolha"] = df[df.escolhida][cols].mean()  # estratégia: versão escolhida pelo cosseno
    print(tab.round(3))
    return df


# ---------- Etapa 5: material final (HTML para revisão humana) ----------
def stage_report(df):
    parts = ["<meta charset='utf-8'><title>Material gerado</title>"
             "<style>body{font-family:sans-serif;max-width:1100px;margin:auto}"
             ".row{display:flex;gap:16px;margin:24px 0}.c{flex:1}img{width:100%}"
             "small{color:#555}</style><h1>Material gerado (revisar antes de usar)</h1>"]
    for sid, g in df.groupby("id", sort=False):
        parts.append("<div class='row'>")
        for _, r in g.iterrows():
            parts.append(
                f"<div class='c'><b>{r.nivel}</b>{' ★ escolhida' if r.escolhida else ''}<br>"
                f"<img src='img/{sid}_{r.nivel}.png'>"
                f"<p>{html.escape(r.frase)}</p>"
                f"<small>Legenda: {html.escape(r.legenda)}<br>"
                f"BERTScore F1 (vs frase): {r.bertscore_f1_vs_frase:.3f}<br>"
                f"Triagem infantil (modelo): {r.infantil}<br>"
                f"Revisão humana: [ ] aprovado [ ] reprovado</small></div>")
        parts.append("</div>")
    (OUT / "relatorio.html").write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="dados/triplets_length.tsv")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed-amostra", type=int, default=SEED, help="seed do sorteio das triplas")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--tau", type=float, default=0.80, help="cosseno mínimo para aceitar uma simplificação")
    ap.add_argument("--ids", nargs="+", help="ids das triplas (ex.: s01996) no lugar do sorteio")
    a = ap.parse_args()
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    log("1/5 base e amostra")
    base = load_base(a.tsv)
    eda(base)
    items = load_items(base, a.n, a.seed_amostra, a.ids)
    log("2/5 similaridade e escolha da versão")
    stage_similarity(items, a.tau)
    (OUT / "amostra.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    log("3/5 geração: prompts (CPU)")
    stage_prompts(items)
    log("3/5 geração: imagens (GPU)")
    stage_images(items, a.size, a.steps)
    log("4/5 legenda e triagem infantil")
    caps = stage_captions(items)
    log("4/5 BERTScore")
    df = stage_metrics(items, caps)
    log("5/5 relatório")
    stage_report(df)
    log("Pronto: saida/relatorio.html e saida/resultados.csv")
