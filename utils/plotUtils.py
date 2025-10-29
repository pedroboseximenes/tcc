import matplotlib.pyplot as plt

def gerar_plot(eixo_x, eixo_y, titulo, xlabel, ylabel, legenda):
    plt.figure(figsize=(12, 6))
    plt.plot(eixo_x)
    plt.plot(eixo_y)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legenda, loc='upper right')
    plt.savefig("pictures/"+titulo +".png")
    plt.close()