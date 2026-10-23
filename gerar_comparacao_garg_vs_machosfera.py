import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.downloader as api

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

PASTA_SAIDA = "./graficos_comparacao_garg_vs_machosfera"
os.makedirs(PASTA_SAIDA, exist_ok=True)

ANOS = [2014, 2015, 2016, 2017, 2018]

# -------------------------------------------------------------------------
# 1. Carregar listas de palavras do Garg
# -------------------------------------------------------------------------
CAMINHO_DATA = "../data/"

def carregar_lista(caminho):
    if not os.path.exists(caminho):
        return []
    with open(caminho, "r", encoding="utf-8") as f:
        return [l.strip().lower() for l in f if l.strip()]

MALE_WORDS = carregar_lista(CAMINHO_DATA + "male_pairs.txt")
FEMALE_WORDS = carregar_lista(CAMINHO_DATA + "female_pairs.txt")
INTEL_ADJS = carregar_lista(CAMINHO_DATA + "adjectives_intelligencegeneral.txt")
APP_ADJS = carregar_lista(CAMINHO_DATA + "adjectives_appearance.txt")

# -------------------------------------------------------------------------
# 2. Carregar o Modelo de Referência de Garg et al. (GloVe Wikipedia / Sociedade Ampla)
# -------------------------------------------------------------------------
print("Carregando embedding de referência de Garg et al. (GloVe Wikipedia - Sociedade Ampla)...")
garg_model = api.load("glove-wiki-gigaword-50")

def get_vec_garg(w):
    if w in garg_model:
        v = garg_model[w]
        return v / np.linalg.norm(v)
    return None

vm_garg = np.mean([get_vec_garg(w) for w in MALE_WORDS if get_vec_garg(w) is not None], axis=0)
vf_garg = np.mean([get_vec_garg(w) for w in FEMALE_WORDS if get_vec_garg(w) is not None], axis=0)
vm_garg /= np.linalg.norm(vm_garg)
vf_garg /= np.linalg.norm(vf_garg)

def calcular_bias_garg(w):
    vw = get_vec_garg(w)
    if vw is None:
        return np.nan
    return float(np.linalg.norm(vw - vm_garg) - np.linalg.norm(vw - vf_garg))

print("Calculando viés de referência no modelo de Garg...")
bias_garg_intelligent = calcular_bias_garg("intelligent")
bias_garg_beautiful = calcular_bias_garg("beautiful")

# Médias dos grupos no Garg
bias_garg_grupo_intel = np.nanmean([calcular_bias_garg(w) for w in INTEL_ADJS])
bias_garg_grupo_app = np.nanmean([calcular_bias_garg(w) for w in APP_ADJS])

# -------------------------------------------------------------------------
# 3. Carregar resultados da Machosfera (Word2Vec, BERT, RoBERTa, ModernBERT)
# -------------------------------------------------------------------------
df_w2v = pd.read_csv("./resultado_poc.csv")
df_w2v["Modelo"] = "Word2Vec (Machosfera)"

df_trans = pd.read_csv("./resultados_transformers/vies_modelos_transformers.csv")
nome_map = {
    "bert": "BERT (Machosfera)",
    "roberta": "RoBERTa (Machosfera)",
    "modernbert": "ModernBERT (Machosfera)"
}
df_trans["Modelo"] = df_trans["Modelo"].map(nome_map)

df_machosfera = pd.concat([df_w2v, df_trans], ignore_index=True)

# Adicionar a 5ª linha: Benchmark Garg et al. (Sociedade Geral) replicada para os anos
linhas_garg = []
for ano in ANOS:
    linhas_garg.append({
        "Modelo": "Garg et al. (Sociedade Ampla / Wiki)",
        "Ano": ano,
        "Palavra": "intelligent",
        "Categoria": "adjetivo_inteligencia",
        "Bias": bias_garg_intelligent
    })
    linhas_garg.append({
        "Modelo": "Garg et al. (Sociedade Ampla / Wiki)",
        "Ano": ano,
        "Palavra": "beautiful",
        "Categoria": "adjetivo_aparencia",
        "Bias": bias_garg_beautiful
    })
    # Representação para grupos
    linhas_garg.append({
        "Modelo": "Garg et al. (Sociedade Ampla / Wiki)",
        "Ano": ano,
        "Palavra": "_grupo_intel",
        "Categoria": "adjetivo_inteligencia",
        "Bias": bias_garg_grupo_intel
    })
    linhas_garg.append({
        "Modelo": "Garg et al. (Sociedade Ampla / Wiki)",
        "Ano": ano,
        "Palavra": "_grupo_app",
        "Categoria": "adjetivo_aparencia",
        "Bias": bias_garg_grupo_app
    })

df_total = pd.concat([df_machosfera, pd.DataFrame(linhas_garg)], ignore_index=True)

# -------------------------------------------------------------------------
# Paleta de Cores e Estilos de Linha
# -------------------------------------------------------------------------
cores = {
    "Word2Vec (Machosfera)": "#1f77b4",       # Azul
    "BERT (Machosfera)": "#ff7f0e",           # Laranja
    "RoBERTa (Machosfera)": "#2ca02c",        # Verde
    "ModernBERT (Machosfera)": "#d62728",     # Vermelho
    "Garg et al. (Sociedade Ampla / Wiki)": "#000000" # Preto / Linha de Referência
}

estilos = {
    "Word2Vec (Machosfera)": "-",
    "BERT (Machosfera)": "-",
    "RoBERTa (Machosfera)": "-",
    "ModernBERT (Machosfera)": "-",
    "Garg et al. (Sociedade Ampla / Wiki)": "--"
}

# -------------------------------------------------------------------------
# Gráfico 1: Termo 'intelligent' com a 5ª linha do Garg
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sub_intel = df_total[df_total["Palavra"] == "intelligent"]
for mod in sub_intel["Modelo"].unique():
    sub = sub_intel[sub_intel["Modelo"] == mod]
    plt.plot(sub["Ano"], sub["Bias"], label=mod, color=cores.get(mod, "gray"),
             linestyle=estilos.get(mod, "-"), linewidth=3.0 if "Garg" in mod else 2.2,
             marker="o" if "Garg" not in mod else "")

plt.axhline(0, color="gray", linestyle=":", alpha=0.6, label="Neutralidade Absoluta")
plt.title("Comparação: Machosfera vs Sociedade Ampla (Garg et al.)\nTrajetória do termo 'intelligent' (< 0 Homens | > 0 Mulheres)")
plt.xlabel("Ano")
plt.ylabel("Viés Relativo (Relative Norm Distance)")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/garg_vs_machosfera_intelligent.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Gráfico 2: Termo 'beautiful' com a 5ª linha do Garg
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sub_beau = df_total[df_total["Palavra"] == "beautiful"]
for mod in sub_beau["Modelo"].unique():
    sub = sub_beau[sub_beau["Modelo"] == mod]
    plt.plot(sub["Ano"], sub["Bias"], label=mod, color=cores.get(mod, "gray"),
             linestyle=estilos.get(mod, "-"), linewidth=3.0 if "Garg" in mod else 2.2,
             marker="s" if "Garg" not in mod else "")

plt.axhline(0, color="gray", linestyle=":", alpha=0.6, label="Neutralidade Absoluta")
plt.title("Comparação: Machosfera vs Sociedade Ampla (Garg et al.)\nTrajetória do termo 'beautiful' (< 0 Homens | > 0 Mulheres)")
plt.xlabel("Ano")
plt.ylabel("Viés Relativo (Relative Norm Distance)")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/garg_vs_machosfera_beautiful.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Gráfico 3: Média do Grupo de Inteligência
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sub_g_intel = df_total[df_total["Categoria"] == "adjetivo_inteligencia"]
for mod in sub_g_intel["Modelo"].unique():
    sub = sub_g_intel[sub_g_intel["Modelo"] == mod].groupby("Ano")["Bias"].mean().reset_index()
    plt.plot(sub["Ano"], sub["Bias"], label=mod, color=cores.get(mod, "gray"),
             linestyle=estilos.get(mod, "-"), linewidth=3.0 if "Garg" in mod else 2.2,
             marker="^" if "Garg" not in mod else "")

plt.axhline(0, color="gray", linestyle=":", alpha=0.6, label="Neutralidade Absoluta")
plt.title("Média dos Adjetivos de Inteligência: Machosfera vs Garg et al.\n(< 0 Homens | > 0 Mulheres)")
plt.xlabel("Ano")
plt.ylabel("Viés Médio")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/garg_vs_machosfera_grupo_inteligencia.png", dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Gráfico 4: Média do Grupo de Aparência Física
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sub_g_app = df_total[df_total["Categoria"] == "adjetivo_aparencia"]
for mod in sub_g_app["Modelo"].unique():
    sub = sub_g_app[sub_g_app["Modelo"] == mod].groupby("Ano")["Bias"].mean().reset_index()
    plt.plot(sub["Ano"], sub["Bias"], label=mod, color=cores.get(mod, "gray"),
             linestyle=estilos.get(mod, "-"), linewidth=3.0 if "Garg" in mod else 2.2,
             marker="D" if "Garg" not in mod else "")

plt.axhline(0, color="gray", linestyle=":", alpha=0.6, label="Neutralidade Absoluta")
plt.title("Média dos Adjetivos de Aparência: Machosfera vs Garg et al.\n(< 0 Homens | > 0 Mulheres)")
plt.xlabel("Ano")
plt.ylabel("Viés Médio")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(f"{PASTA_SAIDA}/garg_vs_machosfera_grupo_aparencia.png", dpi=300)
plt.close()

print(f"\n[SUCESSO] Todos os gráficos comparativos com a 5ª linha de Garg et al. gerados em: {PASTA_SAIDA}/")
