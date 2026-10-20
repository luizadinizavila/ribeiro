import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Configuração de estilo visual
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

PASTA_RESULTADOS = 'graficos_completos_genero'
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

# 1. Carregamento de listas de palavras
def carregar_lista(caminho):
    if not os.path.exists(caminho):
        print(f"Aviso: {caminho} não encontrado.")
        return []
    with open(caminho, 'r', encoding='utf-8') as f:
        return [linha.strip().lower() for linha in f if linha.strip()]

CAMINHO_DATA = '../data/'
male_words = carregar_lista(CAMINHO_DATA + 'male_pairs.txt')
female_words = carregar_lista(CAMINHO_DATA + 'female_pairs.txt')

occupations = carregar_lista(CAMINHO_DATA + 'occupations1950.txt')
occupations_prof = carregar_lista(CAMINHO_DATA + 'occupations1950_professional.txt')
adjectives_all = carregar_lista(CAMINHO_DATA + 'personalitytraits_original.txt')
adjectives_intel = carregar_lista(CAMINHO_DATA + 'adjectives_intelligencegeneral.txt')
adjectives_app = carregar_lista(CAMINHO_DATA + 'adjectives_appearance.txt')

# 2. Carregamento de dados externos (Censo e MTurk)
def carregar_censo():
    df = pd.read_csv(CAMINHO_DATA + 'occupation_percentages_gender_occ1950.csv')
    # calcular log-proportion: log(p / (1 - p))
    df = df[(df['Female'] > 1e-4) & (df['Female'] < 1 - 1e-4)].copy()
    df['log_prop'] = np.log(df['Female'] / (1.0 - df['Female']))
    return df

def carregar_mturk():
    df = pd.read_csv(CAMINHO_DATA + 'mturk_stereotypes.csv')
    return dict(zip(df['occupation'], df['stereotype_score']))

df_censo = carregar_censo()
mturk_dict = carregar_mturk()

# 3. Carregamento de vetores normalizados
def carregar_vetores(ano):
    caminho = f'./vetores_poc/normalized_vectors_{ano}.txt'
    vetores = {}
    if not os.path.exists(caminho):
        return vetores
    with open(caminho, 'r', encoding='utf-8') as f:
        for linha in f:
            partes = linha.strip().split()
            if len(partes) > 1:
                vetores[partes[0]] = np.array([float(x) for x in partes[1:]])
    return vetores

ANOS = [2014, 2015, 2016, 2017, 2018]
dados_vetores = {}
centros_homem = {}
centros_mulher = {}

print("Carregando embeddings por ano...")
for ano in ANOS:
    v = carregar_vetores(ano)
    if v:
        dados_vetores[ano] = v
        vh = [v[p] for p in male_words if p in v]
        vm = [v[p] for p in female_words if p in v]
        centros_homem[ano] = np.mean(vh, axis=0)
        centros_mulher[ano] = np.mean(vm, axis=0)

# Função para calcular Relative Norm Distance (Equação 5 de Garg et al.)
# Bias = ||w - v_male|| - ||w - v_female||
# > 0: Mais associado a Mulheres | < 0: Mais associado a Homens
def calcular_bias_palavra(palavra, ano):
    if ano not in dados_vetores or palavra not in dados_vetores[ano]:
        return np.nan
    v_word = dados_vetores[ano][palavra]
    d_m = np.linalg.norm(v_word - centros_homem[ano])
    d_f = np.linalg.norm(v_word - centros_mulher[ano])
    return d_m - d_f

print("Processando cálculos e gerando os gráficos de Garg et al...")

# ==============================================================================
# FIGURA 1: Fig 1a do Artigo - Dispersão Ocupação no Censo vs Viés no Embedding
# ==============================================================================
print("1. Gerando Figura 1a (Snapshot Ocupações vs Censo)...")
# Usaremos 2015 (ano do snapshot de 2015 do censo do Garg)
censo_2015 = df_censo[df_censo['Census year'] == 2015].copy()
pontos_fig1a = []
for occ in occupations:
    bias = calcular_bias_palavra(occ, 2015)
    row_c = censo_2015[censo_2015['Occupation'] == occ]
    if not np.isnan(bias) and not row_c.empty:
        pontos_fig1a.append({
            'Occupation': occ,
            'Bias': bias,
            'Log_Prop': row_c.iloc[0]['log_prop'],
            'Female_Prop': row_c.iloc[0]['Female']
        })

df_fig1a = pd.DataFrame(pontos_fig1a)
if not df_fig1a.empty:
    plt.figure(figsize=(9, 6))
    sns.regplot(data=df_fig1a, x='Bias', y='Log_Prop',
                scatter_kws={'alpha': 0.6, 'color': '#2b5c8f'},
                line_kws={'color': '#d95f02', 'linewidth': 2})
    
    # Destacar algumas ocupações clássicas
    destaques = ['nurse', 'engineer', 'teacher', 'mechanic', 'dancer', 'carpenter', 'secretary', 'librarian']
    for occ in destaques:
        sub = df_fig1a[df_fig1a['Occupation'] == occ]
        if not sub.empty:
            plt.annotate(occ, (sub['Bias'].values[0], sub['Log_Prop'].values[0]),
                         fontsize=9, weight='bold', xytext=(5, 5), textcoords='offset points')
            
    r, p = pearsonr(df_fig1a['Bias'], df_fig1a['Log_Prop'])
    plt.title(f"Garg Fig 1a Adaptada: Censo (2015) vs Viés de Gênero na Machosfera (2015)\n(r={r:.3f}, r²={r**2:.3f}, p={p:.2e})")
    plt.xlabel("Viés de Gênero no Embedding (< 0 Homens | > 0 Mulheres)")
    plt.ylabel("Log Proporção de Mulheres no Censo 2015")
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{PASTA_RESULTADOS}/fig1a_censo_vs_bias_ocupacoes.png", dpi=300)
    plt.close()

# ==============================================================================
# FIGURA 2: Fig 1b do Artigo - Média de Viés vs Média do Censo ao Longo do Tempo
# ==============================================================================
print("2. Gerando Figura 1b (Evolução Temporal Ocupações vs Censo)...")
# Ocupações presentes em todos os anos analisados
occs_consistentes = []
for occ in occupations:
    tem_todos = all(not np.isnan(calcular_bias_palavra(occ, a)) for a in ANOS)
    if tem_todos:
        occs_consistentes.append(occ)

media_bias_ano = []
media_censo_ano = []
for ano in ANOS:
    biases = [calcular_bias_palavra(occ, ano) for occ in occs_consistentes]
    media_bias_ano.append(np.mean(biases))
    # pegar proporção média do censo no ano mais próximo (censo disponível vai até 2015)
    ano_censo = min(ano, 2015)
    censo_sub = df_censo[(df_censo['Census year'] == ano_censo) & (df_censo['Occupation'].isin(occs_consistentes))]
    media_censo_ano.append(censo_sub['log_prop'].mean() if not censo_sub.empty else np.nan)

fig, ax1 = plt.subplots(figsize=(9, 5))
color = '#1f77b4'
ax1.set_xlabel('Ano')
ax1.set_ylabel('Viés Médio de Mulheres no Embedding', color=color)
l1 = ax1.plot(ANOS, media_bias_ano, color=color, marker='o', linewidth=2.5, label='Viés Embedding (Mulheres)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#2ca02c'
ax2.set_ylabel('Log Proporção Média de Mulheres no Censo', color=color)
l2 = ax2.plot(ANOS, media_censo_ano, color=color, marker='s', linestyle='--', linewidth=2, label='Censo Real (Log-Prop)')
ax2.tick_params(axis='y', labelcolor=color)

linhas = l1 + l2
labels = [l.get_label() for l in linhas]
ax1.legend(linhas, labels, loc='upper left')

plt.title("Garg Fig 1b Adaptada: Trajetória Temporal de Ocupações na Machosfera vs Censo")
plt.tight_layout()
plt.savefig(f"{PASTA_RESULTADOS}/fig1b_trajetoria_ocupacoes_vs_tempo.png", dpi=300)
plt.close()

# ==============================================================================
# FIGURA 3: Fig 2b do Artigo - Matriz de Correlação Temporal de Adjetivos (Phase Shift)
# ==============================================================================
print("3. Gerando Figura 2b (Matriz de Correlação de Adjetivos / Phase Shift)...")
# Lista de adjetivos presentes em todos os anos
adjs_validos = [adj for adj in adjectives_all if all(not np.isnan(calcular_bias_palavra(adj, a)) for a in ANOS)]

matriz_corr = np.zeros((len(ANOS), len(ANOS)))
for i, a1 in enumerate(ANOS):
    v1 = [calcular_bias_palavra(adj, a1) for adj in adjs_validos]
    for j, a2 in enumerate(ANOS):
        v2 = [calcular_bias_palavra(adj, a2) for adj in adjs_validos]
        r_val, _ = pearsonr(v1, v2)
        matriz_corr[i, j] = r_val

plt.figure(figsize=(7, 6))
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=ANOS, yticklabels=ANOS, cbar_kws={'label': 'Correlação de Pearson'})
plt.title("Garg Fig 2b Adaptada: Matriz de Correlação de Adjetivos entre Anos na Machosfera\n(Identificação de Mudança de Fase / Phase Shift)")
plt.xlabel("Ano")
plt.ylabel("Ano")
plt.tight_layout()
plt.savefig(f"{PASTA_RESULTADOS}/fig2b_matriz_correlacao_phase_shift.png", dpi=300)
plt.close()

# ==============================================================================
# FIGURA 4: Adjetivos de Inteligência vs Aparência Física (Seção 3.2 de Garg)
# ==============================================================================
print("4. Gerando Gráfico de Adjetivos: Inteligência vs Aparência Física...")
intel_valid = [w for w in adjectives_intel if all(not np.isnan(calcular_bias_palavra(w, a)) for a in ANOS)]
app_valid = [w for w in adjectives_app if all(not np.isnan(calcular_bias_palavra(w, a)) for a in ANOS)]

media_intel = [np.mean([calcular_bias_palavra(w, a) for w in intel_valid]) for a in ANOS]
media_app = [np.mean([calcular_bias_palavra(w, a) for w in app_valid]) for a in ANOS]

plt.figure(figsize=(9, 5))
plt.plot(ANOS, media_intel, marker='o', linewidth=2.5, color='#1f77b4', label=f'Adjetivos de Inteligência (n={len(intel_valid)})')
plt.plot(ANOS, media_app, marker='s', linewidth=2.5, color='#d62728', label=f'Adjetivos de Aparência Física (n={len(app_valid)})')
plt.axhline(0, color='black', linestyle='--', alpha=0.6, label='Neutralidade Absoluta')
plt.title("Garg Seção 3.2 Adaptada: Trajetória de Inteligência vs Aparência Física na Machosfera")
plt.xlabel("Ano")
plt.ylabel("Viés Médio (< 0 Homens | > 0 Mulheres)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PASTA_RESULTADOS}/adjetivos_inteligencia_vs_aparencia.png", dpi=300)
plt.close()

# ==============================================================================
# FIGURA 5: Fig 10 do Artigo (Apêndice) - Subconjunto de Ocupações Profissionais
# ==============================================================================
print("5. Gerando Figura 10 (Ocupações Profissionais vs Censo)...")
df_prof = df_fig1a[df_fig1a['Occupation'].isin(occupations_prof)].copy()
if not df_prof.empty:
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df_prof, x='Bias', y='Log_Prop',
                scatter_kws={'alpha': 0.7, 'color': '#6a3d9a'},
                line_kws={'color': '#e31a1c', 'linewidth': 2})
    
    for occ in ['doctor', 'lawyer', 'engineer', 'nurse', 'professor', 'author', 'judge', 'scientist']:
        sub = df_prof[df_prof['Occupation'] == occ]
        if not sub.empty:
            plt.annotate(occ, (sub['Bias'].values[0], sub['Log_Prop'].values[0]),
                         fontsize=9, weight='bold', xytext=(5, 5), textcoords='offset points')
            
    r_p, p_p = pearsonr(df_prof['Bias'], df_prof['Log_Prop'])
    plt.title(f"Garg Fig 10 Adaptada: Ocupações Profissionais vs Censo (2015)\n(r={r_p:.3f}, r²={r_p**2:.3f}, p={p_p:.2e})")
    plt.xlabel("Viés de Gênero no Embedding (< 0 Homens | > 0 Mulheres)")
    plt.ylabel("Log Proporção de Mulheres no Censo 2015")
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{PASTA_RESULTADOS}/fig10_ocupacoes_profissionais.png", dpi=300)
    plt.close()

# ==============================================================================
# FIGURA 6: Seção 3.1 do Artigo - Estereótipos do MTurk vs Viés no Embedding
# ==============================================================================
print("6. Gerando Gráfico Seção 3.1 (Viés do Embedding vs Estereótipos Humanos MTurk)...")
pontos_mturk = []
for occ, score in mturk_dict.items():
    bias = calcular_bias_palavra(occ, 2015)
    if not np.isnan(bias):
        pontos_mturk.append({'Occupation': occ, 'Bias': bias, 'MTurk_Score': score})

df_mturk = pd.DataFrame(pontos_mturk)
if not df_mturk.empty:
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df_mturk, x='Bias', y='MTurk_Score',
                scatter_kws={'alpha': 0.6, 'color': '#ff7f00'},
                line_kws={'color': '#33a02c', 'linewidth': 2})
    r_m, p_m = pearsonr(df_mturk['Bias'], df_mturk['MTurk_Score'])
    plt.title(f"Garg Seção 3.1 Adaptada: Viés na Machosfera vs Estereótipos Humanos (MTurk)\n(r={r_m:.3f}, r²={r_m**2:.3f}, p={p_m:.2e})")
    plt.xlabel("Viés de Gênero no Embedding (< 0 Homens | > 0 Mulheres)")
    plt.ylabel("Pontuação Humana de Estereótipo (MTurk)")
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{PASTA_RESULTADOS}/secao3_1_mturk_stereotypes.png", dpi=300)
    plt.close()

# ==============================================================================
# TABELA 7 / TABELA 12-15: Ranking dos Termos Mais Enviesados
# ==============================================================================
print("7. Gerando Tabela de Ranking Top Homens vs Top Mulheres (Formato Tabela 7 do Garg)...")
linhas_ranking = []
for ano in ANOS:
    biases_occ = [(occ, calcular_bias_palavra(occ, ano)) for occ in occupations if not np.isnan(calcular_bias_palavra(occ, ano))]
    biases_occ.sort(key=lambda x: x[1])
    top_homens_occ = [x[0] for x in biases_occ[:5]]
    top_mulheres_occ = [x[0] for x in biases_occ[-5:]][::-1]

    biases_adj = [(adj, calcular_bias_palavra(adj, ano)) for adj in adjectives_all if not np.isnan(calcular_bias_palavra(adj, ano))]
    biases_adj.sort(key=lambda x: x[1])
    top_homens_adj = [x[0] for x in biases_adj[:5]]
    top_mulheres_adj = [x[0] for x in biases_adj[-5:]][::-1]

    for i in range(5):
        linhas_ranking.append({
            'Ano': ano,
            'Rank': i + 1,
            'Ocupação_Mais_Homem': top_homens_occ[i] if i < len(top_homens_occ) else '',
            'Ocupação_Mais_Mulher': top_mulheres_occ[i] if i < len(top_mulheres_occ) else '',
            'Adjetivo_Mais_Homem': top_homens_adj[i] if i < len(top_homens_adj) else '',
            'Adjetivo_Mais_Mulher': top_mulheres_adj[i] if i < len(top_mulheres_adj) else ''
        })

df_ranking = pd.DataFrame(linhas_ranking)
df_ranking.to_csv(f"{PASTA_RESULTADOS}/tabela_top_genero_por_ano.csv", index=False)

print("\n" + "="*60)
print(f"SUCESSO! Todos os gráficos e tabelas foram gerados em: {PASTA_RESULTADOS}/")
print("="*60)
