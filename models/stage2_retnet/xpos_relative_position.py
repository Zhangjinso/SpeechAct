# Copyright (c) 2022 Microsoft
# Licensed under The MIT License (https://github.com/microsoft/torchscale/blob/main/LICENSE)
import torch
import torch.nn as nn

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    ##对两个输入张量sin和cos，分别进行元素乘以scale的操作，并将结果经过duplicate_interleave函数处理。最终返回的是两个经过相同操作的张量
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    #print('x:',x.shape,cos.shape,rotate_every_two(x).shape,sin.shape)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    def __init__(
        self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = 0.
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]  ###(32,8)  x.shape[1],x.shape[2]/2
        '''
        torch.arange(min_pos, max_pos, 1) 生成一个从min_pos到max_pos，以步长1递增的一维张量（即一个等差数列）。
        .to(self.scale) 将生成的等差数列转换到self.scale张量的数据类型。
        .div(self.scale_base) 将生成的等差数列除以self.scale_base张量的值。
        [:, None] 将一维张量转换为二维张量，其中每个元素都是一行，以便后续的乘法运算。
        最后，将self.scale乘方上述计算结果，得到一个二维张量，其中每个元素都是self.scale与等差数列各元素的乘方
        tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [0.9976, 0.9981, 0.9985, 0.9988, 0.9991, 0.9994, 0.9996, 0.9998],
        [0.9951, 0.9962, 0.9970, 0.9977, 0.9983, 0.9988, 0.9992, 0.9996],
        [0.9927, 0.9943, 0.9955, 0.9965, 0.9974, 0.9982, 0.9988, 0.9995],
        [0.9903, 0.9924, 0.9940, 0.9954, 0.9966, 0.9976, 0.9985, 0.9993],
        [0.9878, 0.9905, 0.9925, 0.9942, 0.9957, 0.9970, 0.9981, 0.9991],
        [0.9854, 0.9886, 0.9910, 0.9931, 0.9948, 0.9964, 0.9977, 0.9989],
                         ................
        '''
        sin, cos = fixed_pos_embedding(scale) ###(32,8)

        if scale.shape[0] > length:
            #print('I am here')
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x
    
    def forward_reverse(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]
        
        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x
    
# test
if __name__ == "__main__":
    x = torch.eye(4).unsqueeze(0)
    xpos = XPOS(4)
    x_rot = xpos(x)
    # apply reverse
    x_rot_rev = xpos.forward(x)

    print(x_rot @ x_rot_rev.transpose(-1, -2))