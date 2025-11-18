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
        #self.linear1 = nn.Linear(hidden_dim, 32)
        #self.linear2 = nn.Linear(32, output_dim)

    def forward(self, input,hidden=None):
        if hidden is None:
            batch_size = input.size(0)
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=input.device)
            c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=input.device)
        else:
            h0, c0 = hidden
        x, (hn, cn) = self.lstm(input, (h0, c0))
        x = x[:, -1, :]       # pega o último timestep -> (batch, hidden_dim)
        x = self.dropout(x)
        x = self.fc(x)        # (batch, output_dim)
        #x = F.relu(self.linear1(x))
        #x = self.linear2(x)
        return x