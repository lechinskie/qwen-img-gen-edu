"""
Experimento complementar ao main.py: efeito do IDIOMA do prompt e da FONTE das frases.

Cada item de --itens tem: id, cond (condição), lang (idioma do prompt: pt|en), frase, fonte
e, opcionalmente, img (imagem já gerada pelo main.py; não gera de novo, só avalia).

Condições usadas no relatório:
  A  porsimples_pt   frases do PorSimplesSent, prompt em português (imagens do main.py)
  B  porsimples_en   as mesmas frases traduzidas, prompt em inglês
  B2 porsimples_mix  frase em português dentro do template em inglês
  C  historias_en    frases de datasets de histórias infantis, prompt em inglês
  D  historias_pt    as mesmas frases traduzidas, prompt em português

Uso: python experimento.py --itens experimento/itens.json --out experimento/saida
Reaproveita o main.py (NF4, Qwen3-VL, mpnet) e os mesmos parâmetros: 1024x1024, 25 passos, seed 42.
Além da legenda e da triagem infantil, pede ao Qwen3-VL uma nota de fidelidade de 1 a 5
(a frase descreve a imagem?), com a pergunta sempre em inglês para manter o juiz constante.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch

import main as M

PROMPT = {
    "pt": M.PROMPT,
    "en": ("Illustrate the following sentence in a single scene.\n"
           "Sentence: {frase}\n"
           "Style: children's textbook illustration, simple lines, soft colors.\n"
           "Composition: main characters and objects in focus, simple background.\n"
           "Constraints: no written text, no violence or scary elements, "
           "suitable for children aged 6 to 10."),
}
Q_CAPTION = {"pt": M.Q_CAPTION,
             "en": "Describe in one short sentence, in English, what appears in this image."}
Q_KIDS = {"pt": M.Q_KIDS,
          "en": ("Is this image suitable for children aged 6 to 10, with no violence, fear or "
                 "inappropriate content? Answer only YES or NO.")}
Q_JUDGE = ("Does the following sentence describe what this image shows?\n"
           "Sentence: {frase}\n"
           "Answer with a single number from 1 (not at all) to 5 (completely).")


def img_path(it, out):
    return Path(it["img"]) if it.get("img") else out / "img" / f"{it['id']}.png"


def stage_prompts(items, out):
    path = out / "prompt_embeds.pt"
    cache = torch.load(path) if path.exists() else {}
    todo = [it for it in items if not it.get("img") and it["id"] not in cache]
    if not todo:
        return
    from diffusers import QwenImage21Pipeline
    pipe = QwenImage21Pipeline.from_pretrained(M.IMG_MODEL, transformer=None, vae=None, dtype=torch.bfloat16)
    with torch.no_grad():
        for it in todo:
            emb, mask, _ = pipe.encode_prompt(PROMPT[it["lang"]].format(frase=it["frase"]), device="cpu")
            cache[it["id"]] = (emb.cpu(), None if mask is None else mask.cpu())
            M.log(f"  prompt {it['id']}")
    torch.save(cache, path)
    del pipe
    M.free()


def stage_images(items, out, size, steps):
    todo = [it for it in items if not img_path(it, out).exists()]
    if not todo:
        return
    from diffusers import QwenImage21Pipeline
    cache = torch.load(out / "prompt_embeds.pt")
    pipe = QwenImage21Pipeline.from_pretrained(M.IMG_MODEL, text_encoder=None, dtype=torch.bfloat16,
                                               quantization_config=M.nf4("transformer"))
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    for it in todo:
        emb, mask = cache[it["id"]]
        image = pipe(prompt_embeds=emb.to("cuda"),
                     prompt_embeds_mask=None if mask is None else mask.to("cuda"),
                     width=size, height=size, num_inference_steps=steps,
                     generator=torch.Generator("cuda").manual_seed(M.SEED)).images[0]
        image.save(img_path(it, out))
        M.log(f"  imagem {it['id']}")
    del pipe
    M.free()


def stage_captions(items, out):
    path = out / "captions.json"
    caps = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [it for it in items if it["id"] not in caps]
    if not todo:
        return caps
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        M.VL_MODEL, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    proc = AutoProcessor.from_pretrained(M.VL_MODEL)
    for it in todo:
        img = Image.open(img_path(it, out)).convert("RGB")
        kids = M.ask_vl(model, proc, img, Q_KIDS[it["lang"]], 5).upper()
        juiz = M.ask_vl(model, proc, img, Q_JUDGE.format(frase=it["frase"]), 5)
        m = re.search(r"[1-5]", juiz)
        caps[it["id"]] = {
            "legenda": M.ask_vl(model, proc, img, Q_CAPTION[it["lang"]], 80),
            "infantil": "SIM" if kids.startswith(("SIM", "YES")) else "NAO",
            "juiz_1a5": int(m.group()) if m else None,
            "juiz_bruto": juiz,
        }
        M.log(f"  legenda {it['id']}: juiz={caps[it['id']]['juiz_1a5']}")
        path.write_text(json.dumps(caps, ensure_ascii=False, indent=1), encoding="utf-8")
    del model
    M.free()
    return caps


def stage_metrics(items, caps, out):
    st = M.sbert()
    cap = [caps[it["id"]]["legenda"] for it in items]
    frase = [it["frase"] for it in items]
    cos = st.similarity(st.encode(cap, convert_to_tensor=True),
                        st.encode(frase, convert_to_tensor=True)).diagonal()
    df = pd.DataFrame([{
        "id": it["id"], "cond": it["cond"], "lang": it["lang"], "fonte": it.get("fonte", ""),
        "nivel": it.get("nivel", "original"), "frase": it["frase"], "legenda": caps[it["id"]]["legenda"],
        "infantil": caps[it["id"]]["infantil"], "juiz_1a5": caps[it["id"]]["juiz_1a5"],
        "cos_legenda_vs_frase": round(float(cos[i]), 4), "n_palavras": len(it["frase"].split()),
        "img": str(img_path(it, out)),
    } for i, it in enumerate(items)])
    df.to_csv(out / "resultados.csv", index=False, encoding="utf-8-sig")
    tab = df.groupby("cond").agg(n=("id", "size"), juiz=("juiz_1a5", "mean"),
                                 cos=("cos_legenda_vs_frase", "mean"),
                                 infantil=("infantil", lambda s: (s == "SIM").mean()))
    print(tab.round(3))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--itens", default="experimento/itens.json")
    ap.add_argument("--out", default="experimento/saida")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--so-gerar", action="store_true", help="para depois das imagens (sem legenda/métricas)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(a.itens).read_text(encoding="utf-8"))
    M.log(f"{len(items)} itens; {sum(1 for i in items if not i.get('img'))} a gerar")
    M.log("1/4 prompts (CPU)")
    stage_prompts(items, out)
    M.log("2/4 imagens (GPU)")
    stage_images(items, out, a.size, a.steps)
    if a.so_gerar:
        M.log("Pronto (só geração)")
        raise SystemExit
    M.log("3/4 legenda, triagem e juiz (Qwen3-VL)")
    caps = stage_captions(items, out)
    M.log("4/4 métricas")
    stage_metrics(items, caps, out)
    M.log(f"Pronto: {out}/resultados.csv")
