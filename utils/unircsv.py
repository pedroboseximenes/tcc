import pandas as pd
import glob
import os
import numpy as np
import re

# --- CONFIGURAÇÕES ---
# Caminho onde estão as pastas exec_0, exec_1, etc.
# Ajuste aqui se sua pasta raiz tiver outro nome (ex: results/dataset/code_carbon)
nome_diretorio="results_apenas_chuva"
base_dados = "BRDWGD"
#base_dados = "MERGE"

DIRETORIO_RAIZ = f"../../{nome_diretorio}/{base_dados}/code_carbon" 
DIRETORIO_SAIDA = F"../../{nome_diretorio}/{base_dados}/code_carbon"

# Cria a pasta de saída se não existir
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
dados_por_modelo = {}

print(f"Iniciando varredura em: {DIRETORIO_RAIZ}")

# 1. LOOP PELAS 10 EXECUÇÕES
for i in range(10):
    pasta_exec = os.path.join(DIRETORIO_RAIZ, f"exec_{i}")
    
    # Verifica se a pasta existe antes de tentar ler
    if not os.path.exists(pasta_exec):
        print(f"Pasta não encontrada: {pasta_exec} (Pulando...)")
        continue

    # Pega todos os CSVs dentro dessa pasta de execução
    arquivos_csv = glob.glob(os.path.join(pasta_exec, f"*.csv"))
    
    for arquivo in arquivos_csv:
        try:
            # 2. LER O ARQUIVO
            df = pd.read_csv(arquivo)
            
            # Adiciona coluna para sabermos de qual execução veio essa linha
            df['iteracao_ref'] = i 
            nome_arquivo = os.path.basename(arquivo)
            
            prefixo = f"Exec{i}_"
            sufixo = ".csv"
            
            nome_modelo = nome_arquivo.replace(prefixo, "").replace(sufixo, "")
            # 4. AGRUPAR NA MEMÓRIA
            if nome_modelo not in dados_por_modelo:
                dados_por_modelo[nome_modelo] = []
            
            dados_por_modelo[nome_modelo].append(df)
            
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")

# 5. CONCATENAR E SALVAR UM ARQUIVO POR MODELO
print("\n--- Salvando Arquivos Consolidados ---")

if not dados_por_modelo:
    print("Nenhum dado encontrado! Verifique o caminho da pasta.")

for nome_modelo, lista_dfs in dados_por_modelo.items():
    df_final = pd.concat(lista_dfs, ignore_index=True)
    df_final = df_final.sort_values(by='iteracao_ref')
    df_final['iteracao_ref'] = df_final['iteracao_ref'].astype(str)

    # --- CÁLCULO DAS ESTATÍSTICAS ---
    cols_numericas = df_final.select_dtypes(include=[np.number])
    stats = cols_numericas.agg(['mean', 'std'], numeric_only=True)
    
    # O resultado de 'stats' tem o index como 'mean' e 'std'. Vamos resetar.
    df_stats = stats.reset_index() # Agora 'mean'/'std' viram uma coluna chamada 'index'
    
    # Vamos renomear essa coluna 'index' para 'iteracao_ref' para alinhar com o DF principal
    df_stats = df_stats.rename(columns={'index': 'iteracao_ref'})
    
    # Ajusta os nomes para ficar bonito no CSV (ex: 'mean' -> 'MEDIA', 'std' -> 'DESVIO_PADRAO')
    df_stats['iteracao_ref'] = df_stats['iteracao_ref'].replace({
        'mean': 'MEDIA_GERAL', 
        'std': 'DESVIO_PADRAO'
    })
    df_com_estatisticas = pd.concat([df_final, df_stats], ignore_index=True)
    caminho_salvar = os.path.join(DIRETORIO_SAIDA, f"{nome_modelo}_emissions.csv")
    
    df_com_estatisticas.to_csv(caminho_salvar, index=False)
    print(f"Salvo: {caminho_salvar} ({len(df_final)} registros)")

print("\nProcesso concluído!")