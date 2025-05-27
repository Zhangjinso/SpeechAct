import torch
import math
import torch.nn as nn
import numpy as np

import os, sys
sys.path.append(os.getcwd())
from models.modules import ResnetBlock, AttnBlock, Upsample, nonlinearity,AttnBlock

class Decoder(nn.Module):
    def __init__(self, 
                 ch, in_ch, out_ch, ch_mult, num_res_blocks, resolution,
                 attn_resolutions, latent_size=32, dropout = 0.0, resamp_with_conv = True, give_pre_end = False,
                 window_size = 2):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.temb_ch = 0
        self.ch = ch
        self.give_pre_end = give_pre_end

        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        #print('curr_res:',curr_res)
        self.z_shape = (1,in_ch,curr_res,curr_res)
        #print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        self.conv_in = torch.nn.Conv1d(in_ch, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        
        self.norm_out = nn.LayerNorm(block_in)
        self.conv_out = torch.nn.Conv1d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        
      
    def forward(self, h, grain_indices):
        

          temb = None
          h = self.conv_in(h)
  
          h = self.mid.block_1(h, temb)
          h = self.mid.attn_1(h)
          h = self.mid.block_2(h, temb)
         

          for i_level in reversed(range(self.num_resolutions)):
              for i_block in range(self.num_res_blocks+1):
                  h = self.up[i_level].block[i_block](h, temb)
                  if len(self.up[i_level].attn) > 0:
                      h = self.up[i_level].attn[i_block](h)
              if i_level != 0:
                  h = self.up[i_level].upsample(h)
              
  
          if self.give_pre_end:
              return h
          h = self.norm_out(h.transpose(1,2)).transpose(1,2)
          h = nonlinearity(h)
          h = self.conv_out(h)
 
          return h