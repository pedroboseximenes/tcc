import torch.nn as nn
import torch
import torch.nn.functional as F

# CÉLULA 1: DEFINIÇÃO DO MODELO CORRIGIDO

class LstmModel(nn.Module):
    # Definimos os tamanhos como argumentos para maior flexibilidade
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim , drop_rate):
        super(LstmModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim,   # dimensão da entrada (ex: 1 se é série univariada)
                             hidden_dim, # tamanho do "estado escondido"
                             layer_dim, # número de camadas LSTM empilhadas
                                batch_first=True)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, input):
        batch_size = input.size(0)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=input.device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=input.device)

        x, (hn, cn) = self.lstm(input, (h0.detach(), c0.detach()))
        x = x[:, -1, :]       # pega o último timestep -> (batch, hidden_dim)
        x = self.dropout(x)
        x = self.fc(x)        # (batch, output_dim)

        return x