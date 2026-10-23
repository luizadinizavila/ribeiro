import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

PASTA_SAIDA = "./graficos_comparativos_modelos"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# 1. Carregar resultados do Word2Vec
caminho_w2v = "./resultado_poc.csv"
if os.path.exists(caminho_w2v):
    df_w2v = pd.read_csv(caminho_w2v)
    df_w2v["Modelo"] = "word2vec"
else:
    df_w2v = pd.DataFrame()

# 2. Carregar resultados dos Transformers
caminho_trans = "./resultados_transformers/vies_modelos_transformers.csv"
if os.path.exists(caminho_trans):
    df_trans = pd.read_csv(caminho_trans)
else:
    df_trans = pd.DataFrame()

df_total = pd.concat([df_w2v, df_trans], ignore_index=True)

if df_total.empty:
    print("Nenhum dado encontrado para gerar comparação!")
    exit()

print(f"Total de registros carregados: {len(df_total)}")
print("Modelos presentes:", df_total["Modelo"].unique())

# -------------------------------------------------------------------------
# Gráfico 1: Comparativo da palavra 'intelligent'
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
sub_intel = df_total[df_total["Palavra"] == "intelligent"]
if not sub_intel.empty:
    sns.lineplot(data=sub_intel, x="Ano", y="Bias", hue="Modelo", marker="o", linewidth=2.5)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Neutralidade")
    plt.title("Comparação entre Modelos: Viés do termo 'intelligent' (2014-2018)\n(< 0 Homens | > 0 Mulheres)")
    plt.ylabel("Viés Relativo (Relative Norm Distance)")
    plt.tight_layout()
    plt.savefig(f"{PASTA_SAIDA}/comparativo_intelligent.png", dpi=300)
    plt.close()

# -------------------------------------------------------------------------
# Gráfico 2: Comparativo da palavra 'beautiful'
# -------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
sub_beau = df_total[df_total["Palavra"] == "beautiful"]
if not sub_beau.empty:
    sns.lineplot(data=sub_beau, x="Ano", y="Bias", hue="Modelo", marker="s", linewidth=2.5)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Neutralidade")
    plt.title("Comparação entre Modelos: Viés do termo 'beautiful' (2014-2018)\n(< 0 Homens | > 0 Mulheres)")
    plt.ylabel("Viés Relativo (Relative Norm Distance)")
    plt.tight_layout()
    plt.savefig(f"{PASTA_SAIDA}/comparativo_beautiful.png", dpi=300)
    plt.close()

# -------------------------------------------------------------------------
# Gráfico 3: Média dos Adjetivos de Inteligência por Modelo
# -------------------------------------------------------------------------
sub_cat_intel = df_total[df_total["Categoria"] == "adjetivo_inteligencia"]
if not sub_cat_intel.empty:
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=sub_cat_intel, x="Ano", y="Bias", hue="Modelo", marker="^", linewidth=2.5, errorbar=None)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Neutralidade")
    plt.title("Viés Médio do Grupo de Adjetivos de Inteligência por Modelo\n(< 0 Homens | > 0 Mulheres)")
    plt.ylabel("Viés Médio")
    plt.tight_layout()
    plt.savefig(f"{PASTA_SAIDA}/comparativo_grupo_inteligencia.png", dpi=300)
    plt.close()

# -------------------------------------------------------------------------
# Gráfico 4: Média dos Adjetivos de Aparência por Modelo
# -------------------------------------------------------------------------
sub_cat_app = df_total[df_total["Categoria"] == "adjetivo_aparencia"]
if not sub_cat_app.empty:
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=sub_cat_app, x="Ano", y="Bias", hue="Modelo", marker="D", linewidth=2.5, errorbar=None)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Neutralidade")
    plt.title("Viés Médio do Grupo de Adjetivos de Aparência por Modelo\n(< 0 Homens | > 0 Mulheres)")
    plt.ylabel("Viés Médio")
    plt.tight_layout()
    plt.savefig(f"{PASTA_SAIDA}/comparativo_grupo_aparencia.png", dpi=300)
    plt.close()

print(f"\n[SUCESSO] Gráficos comparativos gerados em: {PASTA_SAIDA}/")
