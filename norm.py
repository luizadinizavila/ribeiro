import os
import csv
import re
import numpy as np

# Função do Garg adaptada para Python 3
def normalize(filename, filename_output):
    countnorm0 = 0
    countnormal = 0
    
    with open(filename_output, 'w', encoding='utf-8', newline='') as fo:
        writer = csv.writer(fo, delimiter=' ')
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ')
            
            for row in reader:
                if not row: continue
                
                rowout = row
                # Limpa a palavra para manter o formato do Garg
                word = re.sub('[^a-z]+', '', row[0].strip().lower())
                rowout[0] = word
                
                # O Garg ignora palavras com menos de 2 letras
                if len(word) < 2: continue
                
                # Extrai os números do vetor ignorando o cabeçalho se houver
                try:
                    vetor = [float(x) for x in row[1:] if len(x) > 0]
                except ValueError:
                    continue # Pula linhas de cabeçalho do Word2Vec
                    
                norm = np.linalg.norm(vetor)
                
                if norm < 1e-2:
                    countnorm0 += 1
                else:
                    countnormal += 1
                    for en in range(1, len(rowout)):
                        if len(rowout[en]) > 0:
                            rowout[en] = str(float(rowout[en]) / norm)
                    writer.writerow(rowout)
                    
    print(f"Normalização concluída: {countnormal} vetores normalizados, {countnorm0} ignorados (norma 0).")

# Executa a normalização para os anos da PoC
for ano in range(2014, 2019):
    input_file = f'./vetores_poc/vectors{ano}.txt'
    output_file = f'./vetores_poc/normalized_vectors_{ano}.txt'
    
    if os.path.exists(input_file):
        print(f"\nNormalizando os vetores do ano {ano}...")
        normalize(input_file, output_file)
    else:
        print(f"\nArquivo {input_file} não encontrado. Pulando...")