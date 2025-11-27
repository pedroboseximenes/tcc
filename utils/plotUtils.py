import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def gerar_plot_dois_eixo(eixo_x, eixo_y, titulo, xlabel, ylabel, legenda, dataset,index):
    plt.figure(figsize=(12, 6))
    plt.plot(eixo_x, color='black')
    plt.plot(eixo_y, color='tab:blue')
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legenda, loc='upper right')
    plt.savefig(f"../results/{dataset}/pictures/exec_{index}/{titulo}.png")
    plt.close()


def gerar_grafico_modelos(y_true, y_pred_arima, y_pred_rf, y_pred_lstm, y_pred_bilstm, titulo, dataset, index):
    plt.figure(figsize=(14, 7))
    # Plotar Real
    eixo_x_datas = y_true.index

    if y_true is not None:
        plt.plot(eixo_x_datas, y_true, label='Real', color='black', linewidth=2, linestyle='-')

    # Plotar Previsões
    if y_pred_arima is not None:
        plt.plot(eixo_x_datas, y_pred_arima, label='ARIMA', color='tab:blue', linestyle='--')

    if y_pred_rf is not None:
        plt.plot(eixo_x_datas, y_pred_rf, label='Random Forest', color='tab:orange', linestyle='--')

    if y_pred_lstm is not None:
        plt.plot(eixo_x_datas, y_pred_lstm, label=f'LSTM', color='tab:green', linestyle='--')

    if y_pred_bilstm is not None:
        plt.plot(eixo_x_datas, y_pred_bilstm, label=f'BiLSTM', color='tab:red', linestyle='--')

    plt.title(f'Comparação de Previsões: {titulo}', fontsize=16)
    plt.xlabel('Tempo', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.savefig(f'../results/{dataset}/pictures/exec_{index}/previsoes_{titulo}.png')