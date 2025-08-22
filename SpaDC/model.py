import torch
from torch import Tensor
import torch.nn as nn

class ConvTower(nn.Module):
    def __init__(self, in_channel, out_channel: int, kernel_size, pool_size=2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),    
            nn.MaxPool1d(kernel_size=pool_size),      
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class SpaDC(nn.Module):
    def __init__(self, n_cells: int, hidden_size=32, seq_len: int = 1344, bias: bool = True):
        super().__init__()

        onehot = torch.cat((
            torch.eye(4),  # A, C, G, T
            torch.zeros(1, 4),  # N
        ), dim=0).float()

        self.onehot = nn.Parameter(onehot, requires_grad=False)
        self.seq_len = seq_len

        current_len = seq_len
        # 1
        self.pre_conv = nn.Sequential(  # input: (batch_size, 4, seq_len)
            nn.Conv1d(4, 288, kernel_size=17, padding=8),
            nn.BatchNorm1d(288),            
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),  # output: (batch_size, 288, 448)           
        )
        current_len = current_len // 3

        # 2
        kernel_nums = [288, 288, 323, 363, 407, 456, 512]
        self.conv_towers = []
        for i in range(1, 7):
            self.conv_towers.append(ConvTower(kernel_nums[i - 1], kernel_nums[i], kernel_size=5))
            current_len = current_len // 2  # 448 -> 224 -> 112 -> 56 -> 28 -> 14 -> 7; (batch_size, 512, 7)
        self.conv_towers = nn.Sequential(*self.conv_towers)

        # 3
        self.post_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),  # (batch_size, 256, 7)
        )

        # 4
        self.flatten = nn.Flatten()  # (batch_size, 1792)

        current_len = current_len * 256

        # 5
        self.dense = nn.Sequential(
            nn.Linear(current_len, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(), 
        )

        # 6
        self.cell_embedding = nn.Linear(hidden_size, n_cells, bias=bias)

    def get_embedding(self):
        return self.cell_embedding.state_dict()['weight']

    def forward(self, sequence: Tensor) -> Tensor:
        """
        sequence: (batch_size, seq_len), one-hot encoded sequence, 0: A, 1: C, 2: G, 3: T, 4: else
        """
        # assert sequence.shape[1] == self.seq_len
        current = self.onehot[sequence.long()].transpose(1, 2)
        current = self.pre_conv(current)
        current = self.conv_towers(current)
        current = self.post_conv(current)
        current = self.flatten(current)
        current = self.dense(current)  # (B, hidden_size)
        logits = self.cell_embedding(current)

        logits = torch.sigmoid(logits)

        return logits, current





