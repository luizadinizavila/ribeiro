import pandas as pd
import re
import os

# 1. Carregar o arquivo parquet
print("Carregando o dataset...")
# Substitua 'seu_dataset.parquet' pelo nome do seu arquivo
df = pd.read_parquet('.\\reddit_filtrado.parquet') 

# Naturalmente decodifica a date column de Unix Timestamp com o .to_datetime
df['year'] = pd.to_datetime(df['date_post'], unit='s', errors='coerce').dt.year

# 2. Filtrar anos de 2014 a 2018
df_filtrado = df[(df['year'] >= 2014) & (df['year'] <= 2018)]

# 3. Pegar uma amostra (vamos usar 5% para a PoC rodar rápido no Word2Vec do seu PC)
df_amostra = df_filtrado.sample(frac=0.05, random_state=42)

# Função básica para limpar o texto (Word2Vec prefere texto em minúsculo e sem pontuação pesada)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # remove tudo que não for letra ou espaço
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Limpando textos e exportando para TXT por ano...")
os.makedirs('dados_por_ano_poc', exist_ok=True)

# 4. Salvar um arquivo .txt por ano (Formato que o pipeline do Garg espera)
for ano in range(2014, 2019):
    textos_ano = df_amostra[df_amostra['year'] == ano]['text_post']
    
    with open(f'dados_por_ano_poc/{ano}.txt', 'w', encoding='utf-8') as f:
        for texto in textos_ano:
            texto_limpo = clean_text(texto)
            if len(texto_limpo) > 10: # Ignorar posts muito curtos ou vazios
                f.write(texto_limpo + '\n')
                
print("Extração concluída!")