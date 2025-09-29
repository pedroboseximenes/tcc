
import torch.nn as nn
import torch
 
# CÉLULA 1: DEFINIÇÃO DO MODELO CORRIGIDO

class LstmModel(nn.Module):
    # Definimos os tamanhos como argumentos para maior flexibilidade
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Camada LSTM que espera o batch na primeira dimensão
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Camada linear para gerar a previsão final
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Inicializa os estados oculto (h0) e de célula (c0)
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Passa os dados pela LSTM
        # lstm_out contém as saídas para cada passo da sequência
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # --- PONTO CRÍTICO DA CORREÇÃO ---
        # Pegamos APENAS a saída do ÚLTIMO passo da sequência.
        # O shape de lstm_out é (batch_size, seq_len, hidden_size).
        # lstm_out[:, -1, :] seleciona o último passo para todos os exemplos no batch.
        last_time_step_output = lstm_out[:, -1, :]
        
        # Passa essa saída pela camada linear para obter a previsão final
        out = self.linear(last_time_step_output)
        return out