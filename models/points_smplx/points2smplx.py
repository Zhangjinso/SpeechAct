import sys
import os

sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from models.modules import ResnetBlock,SeqTranslator
from models.stage2_retnet.xpos_relative_position import XPOS

class Points2Smplx(nn.Module):
    def __init__(self,in_channels,hidden_channel,out_channels,num_classes):
        super().__init__()
        #in_channels = in_channels + 64
        self.id_mlp = nn.Conv1d(num_classes, 64, 1, 1)
        self.linear_embedding = nn.Linear(in_channels+64, in_channels)
        #self.xpos = XPOS(in_channels+1)
        self.decoder = SeqTranslator(in_channels,hidden_channel)
        self.final_out = nn.Conv1d(hidden_channel, out_channels, 1, 1)

    def forward(self, in_spec,id):
        id = id.reshape(id.shape[0], -1, 1).repeat(1, 1, in_spec.shape[2]).to(torch.float32)
        id = self.id_mlp(id)
        spectrogram = torch.cat([in_spec, id], dim=1)
        mid = self.linear_embedding(spectrogram.transpose(1,2)).transpose(1,2)
        mid = self.decoder(mid)
        mid = self.final_out(mid)
        #print('out:',mid.shape)
        return mid

