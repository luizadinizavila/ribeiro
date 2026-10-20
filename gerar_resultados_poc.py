import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar as listas de palavras do Garg (na pasta data/ do repositório)
def carregar_lista(caminho):
    if not os.path.exists(caminho): return []
    with open(caminho, 'r', encoding='utf-8') as f:
        return [linha.strip().lower() for linha in f if linha.strip()]

caminho_data = '../data/' # Ajuste se a pasta data/ estiver em outro lugar
palavras_homens = carregar_lista(caminho_data + 'male_pairs.txt')
palavras_mulheres = carregar_lista(caminho_data + 'female_pairs.txt')

# Vamos usar adjetivos de inteligência e aparência para a PoC
adjetivos = carregar_lista(caminho_data + 'adjectives_intelligencegeneral.txt') + \
            carregar_lista(caminho_data + 'adjectives_appearance.txt')

# 2. Função para ler os vetores normalizados
def carregar_vetores(ano):
    caminho = f'./vetores_poc/normalized_vectors_{ano}.txt'
    vetores = {}
    if not os.path.exists(caminho): return vetores
    with open(caminho, 'r', encoding='utf-8') as f:
        for linha in f:
            partes = linha.strip().split()
            if len(partes) > 1:
                vetores[partes[0]] = np.array([float(x) for x in partes[1:]])
    return vetores

# 3. Calcular os Vieses Ano a Ano (Exata Metodologia do Garg)
anos = range(2014, 2019)
resultados = []

for ano in anos:
    print(f"Calculando viés para o ano {ano}...")
    vetores = carregar_vetores(ano)
    if not vetores: continue
    
    # Achar o centro geométrico (média) dos homens e mulheres
    vetores_h = [vetores[p] for p in palavras_homens if p in vetores]
    vetores_m = [vetores[p] for p in palavras_mulheres if p in vetores]
    
    if not vetores_h or not vetores_m: continue
    
    centro_h = np.mean(vetores_h, axis=0)
    centro_m = np.mean(vetores_m, axis=0)
    
    # Calcular a distância para cada adjetivo
    for adj in adjetivos:
        if adj in vetores:
            v_adj = vetores[adj]
            # Equação do Garg: Distância(Adjetivo, Homens) - Distância(Adjetivo, Mulheres)
            # Valores positivos = viés feminino. Valores negativos = viés masculino.
            dist_h = np.linalg.norm(v_adj - centro_h)
            dist_m = np.linalg.norm(v_adj - centro_m)
            bias = dist_h - dist_m
            
            resultados.append({'Ano': ano, 'Palavra': adj, 'Bias': bias})

# 4. Salvar os resultados em CSV para mostrar ao professor
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('resultado_poc.csv', index=False)
print("\nArquivo resultado_poc.csv gerado com sucesso!")

# 5. Opcional: Mostrar um gráfico com 3 palavras de exemplo (Prova visual para o professor)
palavras_exemplo = ['intelligent', 'beautiful', 'rational']
plt.figure(figsize=(10, 5))

for palavra in palavras_exemplo:
    df_palavra = df_resultados[df_resultados['Palavra'] == palavra]
    if not df_palavra.empty:
        plt.plot(df_palavra['Ano'], df_palavra['Bias'], marker='o', label=palavra)

plt.axhline(0, color='black', linestyle='--', label='Neutralidade')
plt.title('Evolução do Viés de Gênero na Machosfera (2014-2018)')
plt.ylabel('Viés (Valores Positivos = Mais ligado a Mulheres)')
plt.xlabel('Ano')
plt.legend()
plt.grid(True)
plt.savefig('grafico_poc.png')
print("Gráfico grafico_poc.png gerado com sucesso!")