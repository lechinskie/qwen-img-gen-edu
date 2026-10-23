"""Contagens por palavra-chave sobre as 797 triplas elegíveis do PorSimplesSent (Seção 2.2 do relatório)."""
import csv
import pandas as pd

df = pd.read_csv("dados/triplets_length.tsv", sep="\t", quoting=csv.QUOTE_NONE, encoding="utf-8")
el = df[(df.changed_ori_nat == "S") & (df.changed_nat_str == "S")]
el = el[el.original_text.str.split().str.len().between(8, 30)]
t = el.original_text.str.lower()
ADULTO = (r"governo|presidente|ministro|deputad|senad|partido|eleiç|polícia|policial|crime|assassin|morte|morreu|"
          r"mortos|doença|vírus|câncer|aids|hiv|sífilis|droga|econom|inflação|dólar|real\b|bilh|milhões de reais|"
          r"imposto|tribunal|juiz|prisão|preso|guerra|ataque|bomba|violên|estupro|tráfico|cigarro|álcool|suicíd")
CITACAO = r'"|disse|afirm|declar|segundo |explica|conta |diz |lembra |garante'
CONCRETO = (r"animal|bicho|cachorro|gato|pássaro|ave|peixe|árvore|flor|planta|parque|escola|casa|rio|mar|praia|sol|"
            r"lua|estrela|chuva|neve|carro|bola|brinquedo|criança|menino|menina|cavalo|vaca|macaco|pinguim|urso|"
            r"leão|elefant|tartaruga|cobra|sapo|abelha|borboleta|formiga|jardim|floresta|montanha|lago")
print("elegíveis:", len(el))
print("tema adulto (palavra-chave):", int(t.str.contains(ADULTO).sum()))
print("fala relatada / citação:", int(t.str.contains(CITACAO).sum()))
print("ao menos um dos dois:", int((t.str.contains(ADULTO) | t.str.contains(CITACAO)).sum()))
print("com substantivo concreto típico:", int(t.str.contains(CONCRETO).sum()))
print("textos de ciência (production_id >= 116):", int((el.production_id >= 116).sum()))
print("\n8 frases sorteadas (seed 0):")
for s in el.original_text.sample(8, random_state=0):
    print(" -", s)
