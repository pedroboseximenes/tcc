import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def gerar_plot_dois_eixo(eixo_x, eixo_y, titulo, xlabel, ylabel, legenda):
    plt.figure(figsize=(12, 6))
    plt.plot(eixo_x)
    plt.plot(eixo_y)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legenda, loc='upper right')
    plt.savefig("pictures/"+titulo +".png")
    plt.close()


def gerar_grafico_unico(imgRandomForest, imgArima, imgLSTM, imgBiLSTM):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    imgs = [
        ("ARIMA",       imgArima),
        ("RandomForest",imgRandomForest),
        ("LSTM",        imgLSTM),
        ("BiLSTM",      imgBiLSTM),
    ]

    for ax, (titulo, caminho) in zip(axs.ravel(), imgs):
        img = mpimg.imread(caminho)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(titulo)

    plt.tight_layout()
    plt.show()
# gerar_grafico_unico("../../resultsSoChuva/TEST [0] - arimaBRDWGD_result.png", 
#                     "../../resultsSoChuva/TEST [0] - lstmRandomForest_BRDWGD_result.png",
#                     "../../resultsSoChuva/TEST [1] - lstmBRDWGD_lookback=30_neuronios=64_camada=2_lr=0.001_droprate=0.5.png",
#                     "../../resultsSoChuva/TEST [2] - BILSTMBRDWGD_lookback=30_neuronios=256_camada=2_lr=0.001_droprate=0.5.png"
#                     )