
import sys
import os

import pickle
import scipy
from models.audio_encoder.wav2vec import Wav2Vec2Model
from models.utils import *
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
import math
from models.modules import SeqTranslator,AttnBlock,ResnetBlock
from models.stage2_retnet.xpos_relative_position import XPOS


        
class AudioEncoder(nn.Module):
    def __init__(self, in_dim,out_dim, identity=True, num_classes=0):
        super().__init__()
        self.identity = identity
        #print('num_classes:',num_classes)
        if self.identity:
            in_dim = in_dim + 64
            self.id_mlp = nn.Conv1d(num_classes, 64, 1, 1)
        
        self.dropout = nn.Dropout(0.1)
        ###add
        self.encoder_linear_embedding = nn.Linear(in_dim, out_dim)
        #self.encoder_xpos = XPOS(out_dim)
        self.encoder = SeqTranslator(out_dim, out_dim)
        
    def forward(self, spectrogram, pre_state=None, id=None, time_steps=None):
        #print('id:',id.shape)
        spectrogram = spectrogram
        spectrogram = self.dropout(spectrogram)
        if self.identity:
            id = id.reshape(id.shape[0], -1, 1).repeat(1, 1, spectrogram.shape[2]).to(torch.float32)
            id = self.id_mlp(id)
            spectrogram = torch.cat([spectrogram, id], dim=1)

        x1 = self.encoder_linear_embedding(spectrogram.transpose(1,2)).transpose(1,2)
        #x1 = self.encoder_xpos(x1).transpose(1,2)
        x1 = self.encoder(x1)

        return x1



class Generator(nn.Module):
    def __init__(self,
                 n_poses,
                 each_dim: list,
                 training=False,
                 device=None,
                 identity=True,
                 num_classes=0,
                 ):
        super().__init__()

        self.training = training
        self.device = device
        self.gen_length = n_poses
        self.identity = identity

        in_dim = 256
        out_dim = 256

        self.wavlm_model = wavlm_init()
        self.audio_feature_map = nn.Linear(1024, in_dim)
        
        self.audio_middle = AudioEncoder(in_dim,out_dim,num_classes=num_classes)
        
        self.decoder_linear_embedding = nn.Linear(out_dim, out_dim)
        
        self.decoder = SeqTranslator(out_dim, out_dim)

        self.final_out =nn.Conv1d(out_dim, each_dim[0]+each_dim[1], 1, 1)

    def forward(self, in_spec, gt_poses=None, id=None, pre_state=None, time_steps=None):
        if self.training:
            time_steps = gt_poses.shape[1]
        
        
        hidden_states= get_wavlm(self.wavlm_model,in_spec.squeeze(),time_steps)
        #print('hidden_states:',hidden_states.shape)
        feature = self.audio_feature_map(hidden_states).transpose(1, 2)
        
        feature = self.audio_middle(feature, id=id)
        out = self.decoder_linear_embedding(feature.transpose(1,2)).transpose(1,2)
       
        out = self.decoder(out)
        out = self.final_out(out).transpose(1, 2)
        return out, None




