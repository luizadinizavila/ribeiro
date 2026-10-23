import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# -------------------------------------------------------------------------
# Mapeamento dos modelos suportados
# -------------------------------------------------------------------------
MODEL_MAP = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "modernbert": "answerdotai/ModernBERT-base"
}

# -------------------------------------------------------------------------
# Carregamento de Listas de Palavras do Garg
# -------------------------------------------------------------------------
CAMINHO_DATA = "../data/"

def carregar_lista(caminho):
    if not os.path.exists(caminho):
        return []
    with open(caminho, "r", encoding="utf-8") as f:
        return [l.strip().lower() for l in f if l.strip()]

MALE_WORDS = carregar_lista(CAMINHO_DATA + "male_pairs.txt")
FEMALE_WORDS = carregar_lista(CAMINHO_DATA + "female_pairs.txt")
OCCUPATIONS = carregar_lista(CAMINHO_DATA + "occupations1950.txt")
INTEL_ADJS = carregar_lista(CAMINHO_DATA + "adjectives_intelligencegeneral.txt")
APP_ADJS = carregar_lista(CAMINHO_DATA + "adjectives_appearance.txt")

TODAS_PALAVRAS_ALVO = set(MALE_WORDS + FEMALE_WORDS + OCCUPATIONS + INTEL_ADJS + APP_ADJS)

# -------------------------------------------------------------------------
# Extração de Embeddings Contextuais por Amostragem
# -------------------------------------------------------------------------
def extrair_embeddings_transformer(model_name_key, ano, max_amostras_por_termo=40, batch_size=16):
    """
    Para cada palavra-alvo (homem, mulher, adjetivo, ocupação), localiza frases
    no corpus daquele ano onde ela ocorre e extrai a representação contextual média
    usando a última camada oculta do Transformer.
    """
    model_hf_id = MODEL_MAP[model_name_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Modelo: {model_name_key.upper()}] Carregando {model_hf_id} no dispositivo: {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_hf_id)
    model = AutoModel.from_pretrained(model_hf_id)
    model.to(device)
    model.eval()

    caminho_corpus = f"./dados_por_ano_poc/{ano}.txt"
    if not os.path.exists(caminho_corpus):
        print(f"Erro: Arquivo do corpus {caminho_corpus} não encontrado!")
        return {}

    # 1. Carregar frases que contenham as palavras de interesse
    print(f"Indexando ocorrências no corpus de {ano}...")
    ocorrencias = {w: [] for w in TODAS_PALAVRAS_ALVO}

    with open(caminho_corpus, "r", encoding="utf-8") as f:
        for linha in f:
            tokens = set(linha.strip().lower().split())
            intersecao = tokens.intersection(TODAS_PALAVRAS_ALVO)
            for w in intersecao:
                if len(ocorrencias[w]) < max_amostras_por_termo:
                    ocorrencias[w].append(linha.strip())

    # 2. Extrair embeddings contextuais para cada palavra
    vetores_palavras = {}
    print(f"Extraindo representações contextuais para {len(TODAS_PALAVRAS_ALVO)} palavras-alvo...")

    with torch.no_grad():
        for palavra, frases in ocorrencias.items():
            if not frases:
                continue

            embeddings_da_palavra = []
            
            # Processar em lotes (batches)
            for i in range(0, len(frases), batch_size):
                lote_frases = frases[i:i + batch_size]
                inputs = tokenizer(
                    lote_frases,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(device)

                outputs = model(**inputs)
                # Last hidden state: (batch_size, seq_len, hidden_dim)
                last_hidden = outputs.last_hidden_state

                # Localizar os tokens correspondentes à palavra no lote
                for idx_frase, frase in enumerate(lote_frases):
                    tokens_id = tokenizer.encode(palavra, add_special_tokens=False)
                    input_ids = inputs["input_ids"][idx_frase].tolist()

                    # Procurar a posição do token da palavra na sequência
                    match_pos = -1
                    for pos in range(len(input_ids) - len(tokens_id) + 1):
                        if input_ids[pos:pos + len(tokens_id)] == tokens_id:
                            match_pos = pos
                            break

                    if match_pos != -1:
                        # Vetor do subtoken da palavra
                        vec = last_hidden[idx_frase, match_pos, :].cpu().numpy()
                    else:
                        # Fallback: média do pooling da frase
                        mask = inputs["attention_mask"][idx_frase].unsqueeze(-1)
                        vec = ((last_hidden[idx_frase] * mask).sum(dim=0) / mask.sum(dim=0)).cpu().numpy()

                    embeddings_da_palavra.append(vec)

            if embeddings_da_palavra:
                # Média agregada das representações contextuais
                vetor_medio = np.mean(embeddings_da_palavra, axis=0)
                # Normalização unitária L2 (conforme Garg et al.)
                norma = np.linalg.norm(vetor_medio)
                if norma > 0:
                    vetores_palavras[palavra] = vetor_medio / norma

    return vetores_palavras

# -------------------------------------------------------------------------
# Cálculo da Relative Norm Distance de Garg et al.
# -------------------------------------------------------------------------
def calcular_viés_ano(vetores):
    vetores_m = [vetores[w] for w in MALE_WORDS if w in vetores]
    vetores_f = [vetores[w] for w in FEMALE_WORDS if w in vetores]

    if not vetores_m or not vetores_f:
        return {}

    centro_m = np.mean(vetores_m, axis=0)
    centro_f = np.mean(vetores_f, axis=0)

    # Normalizar centróides conforme Garg
    centro_m /= np.linalg.norm(centro_m)
    centro_f /= np.linalg.norm(centro_f)

    resultados_bias = {}
    for w, v in vetores.items():
        d_m = np.linalg.norm(v - centro_m)
        d_f = np.linalg.norm(v - centro_f)
        resultados_bias[w] = float(d_m - d_f)

    return resultados_bias

# -------------------------------------------------------------------------
# Execução Principal
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extração de Viés com Transformers (BERT, RoBERTa, ModernBERT)")
    parser.add_argument("--models", nargs="+", default=["bert", "roberta", "modernbert"],
                        help="Modelos para rodar: bert, roberta, modernbert")
    parser.add_argument("--years", nargs="+", type=int, default=[2014, 2015, 2016, 2017, 2018],
                        help="Anos a serem processados")
    parser.add_argument("--samples", type=int, default=30,
                        help="Número máximo de frases amostradas por palavra para média contextual")
    args = parser.parse_args()

    os.makedirs("./resultados_transformers", exist_ok=True)

    todas_linhas = []

    for model_key in args.models:
        if model_key not in MODEL_MAP:
            print(f"Modelo {model_key} inválido! Escolha entre: {list(MODEL_MAP.keys())}")
            continue

        print(f"\n=======================================================")
        print(f"  PROCESSANDO MODELO: {model_key.upper()}")
        print(f"=======================================================")

        for ano in args.years:
            print(f"\n--> Ano {ano}...")
            vetores = extrair_embeddings_transformer(model_key, ano, max_amostras_por_termo=args.samples)
            biases = calcular_viés_ano(vetores)

            for palavra, b_val in biases.items():
                categoria = "outro"
                if palavra in OCCUPATIONS:
                    categoria = "ocupacao"
                elif palavra in INTEL_ADJS:
                    categoria = "adjetivo_inteligencia"
                elif palavra in APP_ADJS:
                    categoria = "adjetivo_aparencia"

                todas_linhas.append({
                    "Modelo": model_key,
                    "Ano": ano,
                    "Palavra": palavra,
                    "Categoria": categoria,
                    "Bias": b_val
                })

    df = pd.DataFrame(todas_linhas)
    caminho_csv = "./resultados_transformers/vies_modelos_transformers.csv"
    df.to_csv(caminho_csv, index=False)
    print(f"\n[SUCESSO] Resultados salvos em: {caminho_csv}")

if __name__ == "__main__":
    main()
