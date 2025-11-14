import torch.nn as nn
import torch
 
# CÉLULA 1: DEFINIÇÃO DO MODELO CORRIGIDO

class BiLstmModel(nn.Module):
    # Definimos os tamanhos como argumentos para maior flexibilidade
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(BiLstmModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim,   # dimensão da entrada (ex: 1 se é série univariada)
                             hidden_dim, # tamanho do "estado escondido"
                             layer_dim, # número de camadas LSTM empilhadas
                                batch_first=True,
                                bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x,hidden=None):
        if hidden is None:
            batch_size = x.size(0)
            h0 = torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim, device=x.device)
            c0 = torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim, device=x.device)
        else:
            h0, c0 = hidden

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]       # pega o último timestep -> (batch, hidden_dim)
        out = self.dropout(out)
        out = self.fc(out)
        return out, (hn, cn)