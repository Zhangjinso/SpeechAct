import os
import sys
import numpy as np
np.set_printoptions(threshold=np.inf)
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())


import pytorch_lightning as pl
from models.modules import (Downsample, ResnetBlock, nonlinearity,AttnBlock)

class TripleGrainEncoder(pl.LightningModule):
    def __init__(self, 
        *, 
        ch, 
        ch_mult=(1,2,4,8), 
        num_res_blocks,
        attn_resolutions, 
        dropout=0.0, 
        resamp_with_conv=True, 
        in_channels,
        resolution, 
        z_channels, 
        **ignore_kwargs
        ):
        super().__init__()
        
        self.ch = ch ##256
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels


        self.conv_in = torch.nn.Conv1d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution  
        in_ch_mult = (1,)+tuple(ch_mult) 
        #print('self.num_res_blocks:',self.num_res_blocks,ch)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level] 
            block_out = ch * ch_mult[i_level]  
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:  
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2  
            self.down.append(down)

        block_in_median_grain = block_in // (ch_mult[-1] // ch_mult[-2])
        block_in_finegrain = block_in_median_grain // (ch_mult[-2] // ch_mult[-3])
        
        self.mid_fine = nn.Module()
        self.mid_fine.block_1 = ResnetBlock(in_channels=block_in_finegrain, out_channels=block_in_finegrain, temb_channels=self.temb_ch, dropout=dropout)
        self.mid_fine.attn_1 = AttnBlock(block_in_finegrain)

        self.mid_fine.block_2 = ResnetBlock(in_channels=block_in_finegrain, out_channels=block_in_finegrain, temb_channels=self.temb_ch, dropout=dropout)

        self.norm_out_fine = nn.LayerNorm(block_in_finegrain)
        self.conv_out_fine = torch.nn.Conv1d(block_in_finegrain, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, x_entropy=None):

 
        
        temb = None

        
        hs = [self.conv_in(x)] 
        for i_level in range(self.num_resolutions):
    
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                    
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            if i_level == self.num_resolutions-3:
                h_fine = h
        
        h_fine = self.mid_fine.block_1(h_fine, temb)  
        h_fine = self.mid_fine.attn_1(h_fine)   
        h_fine = self.mid_fine.block_2(h_fine, temb)  

        h_fine = self.norm_out_fine(h_fine.transpose(1,2)).transpose(1,2) 
        h_fine = nonlinearity(h_fine)  
        h_fine = self.conv_out_fine(h_fine)   

        return {"h_triple":h_fine}
        