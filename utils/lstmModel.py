
import torch.nn as nn
 
class LstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=1, batch_first=True)
        self.linear = nn.Linear(128, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x