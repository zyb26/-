# 绝对位置编码
# PE(pos, 2i) = sin(pos/(10000^2i/dmodel))
# PE(pos, 2i+1) = cos(pos/(10000^2i/dmodel))
# 变量: pos 第几个词; i 第几个维度

# 2i 和 2i + 1 代表embedding的第几个维度
# dmodel 代表embedding的总维度

"""
已知维度  0  1  2  3   4   5  6  7   8   8
2i       0     2      4      6      8
2i+1        1     3       5     7        9
2i       0  0  2  2   4   4  6  6   8    8   公式中 2i 和 2i + 1 都对应的 2i --> (i//2) * 2
横排:只和第几个词有关
纵排:一个sin,一个cos 且和第几个维度有关
"""
import time

import torch
import math

# 绝对位置编码
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4):  # 生成位置编码矩阵 max_len字典的字数 d_model字的维度
        super().__init__()
        start = time.time()
        self.pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for j in range(d_model):
                angle = pos / math.pow(10000, (j // 2) * 2 / d_model)
                if j % 2 == 0:
                    self.pe[pos][j] = math.sin(angle)  # 偶数维
                else:
                    self.pe[pos][j] = math.cos(angle)  # 奇数维
        end = time.time()
        print(self.pe)
        print(end - start)

    def forward(self, x):  # 输入[n, t, e]向量
        return x + self.pe[: x.size(1)]  # 根据输入的长度来拿位置编码 会有一个广播机制

if __name__ == '__main__':
    net = PositionalEncoding(5)
    # 模拟输入2句话
    embed = nn.Embedding(10, 5)
    inputs = torch.tensor([[0, 1],
                           [1, 2]])
    e = embed(inputs)
    print(e)
    print(e.size(1))
    print(net.pe[: e.size(1)])
    print(net(e))

