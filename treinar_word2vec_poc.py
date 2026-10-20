import os
from gensim.models import Word2Vec

anos = range(2014, 2019)
tamanho_vetor = 300 # Tamanho padrão usado pelo Garg

os.makedirs('vetores_poc', exist_ok=True)

for ano in anos:
    print(f"Treinando vetores SGNS (Word2Vec) para o ano {ano}...")
    
    # 1. Carregar o arquivo .txt gerado no passo anterior
    caminho_txt = f'dados_por_ano_poc/{ano}.txt'
    if not os.path.exists(caminho_txt):
        print(f"Arquivo {ano}.txt não encontrado. Pulando...")
        continue
        
    # Lê as frases (cada linha é um post da Machosfera)
    with open(caminho_txt, 'r', encoding='utf-8') as f:
        sentences = [line.strip().split() for line in f if line.strip()]
        
    if not sentences:
        continue
        
    # 2. Treinar o modelo SGNS (sg=1 significa Skip-Gram, o mesmo do artigo)
    # min_count=5 ignora palavras muito raras
    modelo = Word2Vec(sentences=sentences, vector_size=tamanho_vetor, window=10, min_count=5, workers=4, sg=1)
    
    # 3. Salvar no formato exato de texto que o pipeline do Garg espera
    caminho_saida = f'vetores_poc/vectors{ano}.txt'
    modelo.wv.save_word2vec_format(caminho_saida, binary=False)
    
    print(f"Vetores do ano {ano} salvos com sucesso!")