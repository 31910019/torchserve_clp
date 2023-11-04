import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model import GELU, PositionalEmbedding, LayerNorm, Attention, MultiHeadedAttention, PositionwiseFeedForward, SublayerConnection, TransformerBlock

class BERT4NILM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.original_len = args.window_size
        self.latent_len = int(self.original_len / 2)
        self.dropout_rate = args.drop_out

        self.hidden = 32
        self.heads = 1
        self.n_layers = 1
        self.output_size = args.output_size

        self.conv = nn.Conv1d(in_channels=1, out_channels=self.hidden,
                               kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.position = PositionalEmbedding(
            max_len=self.latent_len, d_model=self.hidden)
        self.layer_norm = LayerNorm(self.hidden)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            self.hidden, self.heads, self.hidden * 4, self.dropout_rate) for _ in range(self.n_layers)])

        self.deconv = nn.ConvTranspose1d(
            in_channels=self.hidden, out_channels=self.hidden, kernel_size=4, stride=2, padding=1)
        self.linear1 = nn.Linear(self.hidden, 128)
        self.linear2 = nn.Linear(128, self.output_size)

        self.truncated_normal_init()
        print('self.output_size ',self.output_size )

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        params = list(self.named_parameters())
        for n, p in params:
            if 'layer_norm' in n:
                continue
            else:
                with torch.no_grad():
                    l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                    u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
                    p.uniform_(2 * l - 1, 2 * u - 1)
                    p.erfinv_()
                    p.mul_(std * math.sqrt(2.))
                    p.add_(mean)

    def forward(self, sequence):    #torch.Size([128, 480])
        x_token = self.pool(self.conv(sequence.unsqueeze(1))).permute(0, 2, 1)
        embedding = x_token + self.position(sequence)
        x = self.dropout(self.layer_norm(embedding))

        mask = None
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        x = self.deconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)     #torch.Size([128, 480, 1])
        # print('ssss',x.shape)
        return x


# torch-model-archiver --model-name BERT4NILM --version 1.0 --model-file BERT4NILM.py --serialized-file transformer.pt --handler NILMHandler.py
