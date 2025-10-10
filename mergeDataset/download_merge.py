import os
import requests
from datetime import date, timedelta

def baixar_merge_cptec(ano_inicial=1998, ano_final=2025, mes_final=10, pasta_destino="/home/pbose/tcc/dataset/merge/"):
    """
    Baixa arquivos MERGE_CPTEC_<YYYYMMDD>.grib2 do FTP do CPTEC.
    - De 1998 até 2025 (mês 10)
    - Cada arquivo diário
    """

    base_url = "https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY"
    os.makedirs(pasta_destino, exist_ok=True)

    for ano in range(ano_inicial, ano_final + 1):
        # Define o último mês do ano
        ultimo_mes = mes_final if ano == ano_final else 12
        for mes in range(1, ultimo_mes + 1):
            dias_no_mes = 31
            # Ajuste de meses com 30 dias e fevereiro
            if mes in [4, 6, 9, 11]:
                dias_no_mes = 30
            elif mes == 2:
                dias_no_mes = 29 if (ano % 4 == 0 and (ano % 100 != 0 or ano % 400 == 0)) else 28

            for dia in range(1, dias_no_mes + 1):
                nome_arquivo = f"MERGE_CPTEC_{ano}{mes:02d}{dia:02d}.grib2"
                url = f"{base_url}/{ano}/{mes:02d}/{nome_arquivo}"
                caminho_local = os.path.join(pasta_destino, nome_arquivo)

                # Pula se já baixado
                if os.path.exists(caminho_local):
                    print(f"✅ Já existe: {nome_arquivo}")
                    continue

                try:
                    resposta = requests.get(url, timeout=30)
                    if resposta.status_code == 200:
                        with open(caminho_local, "wb") as f:
                            f.write(resposta.content)
                        print(f"⬇️ Baixado: {nome_arquivo}")
                    else:
                        print(f"⚠️ Não encontrado ({resposta.status_code}): {url}")
                except Exception as e:
                    print(f"❌ Erro ao baixar {nome_arquivo}: {e}")

    print("\n✅ Download concluído.")

if __name__ == "__main__":
    baixar_merge_cptec()