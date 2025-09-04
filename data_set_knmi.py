import requests
import os
import h5py

def baixar_arquivos(arquivos):
    for arquivo in arquivos:
        nome_arquivo = arquivo["filename"]
        download_url = f"https://api.dataplatform.knmi.nl/open-data/v1/datasets/radar_forecast/versions/2.0/files/{nome_arquivo}/url"

        print(f"Baixando {nome_arquivo}...")

        download_response = requests.get(download_url, headers=headers)

        if download_response.status_code == 200:
            download_url = download_response.json().get("temporaryDownloadUrl")
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(caminho_completo, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Salvo em: {caminho_completo}")
                f.close()
        else:
            print(f"Erro ao baixar {nome_arquivo}: {download_response.status_code}")

#url antigo = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/nl_rdr_data_rtcor_5m/versions/1.0/files"
url = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/radar_forecast/versions/2.0/files"

api_key = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjYyZjVlNmYxZTExODQzNTJhODYyNDY3MGRlNDQzZTIyIiwiaCI6Im11cm11cjEyOCJ9"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
next_page_token = None
arquivo_nome = {"file_name" :"RAD_NL25_RAC_5min_train_test_atual.h5"}
caminho_completo = os.path.join("/home/pbose/tcc/SmaAt-UNet-master/dataset", arquivo_nome["file_name"])

max_keys = 10
total_dataset = 100
quantidade_baixada = 0
while quantidade_baixada <= total_dataset:
    params =  {"maxKeys": f"{max_keys}", "nextPageToken": next_page_token}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        arquivos = data.get("files", [])
        next_page_token = data.get("nextPageToken")

        baixar_arquivos(arquivos)
        quantidade_baixada += max_keys
    else:
        print(f"Erro ao obter a lista de arquivos: {response.status_code}")
        print(response.text)

with h5py.File(caminho_completo, 'r') as f:
    # List all groups or datasets in the file
    print("Keys:", list(f.keys()))

    # Access a specific dataset by name
    print(f)