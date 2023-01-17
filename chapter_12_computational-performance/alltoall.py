import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import operations as ops
from mindspore.communication import init, get_rank, get_group_size

class AlltoAll(nn.Cell):
    def __init__(self, split_count=1):
        super().__init__()
        self.alltoall = ops.AlltoAll(split_count=split_count, split_dim=0, concat_dim=0)
        self.split = ops.Split(axis=0, output_num=split_count)

    def construct(self, inputs):
        return self.split(self.alltoall(inputs))[0]

init()
group_size = get_group_size()
data = mnp.arange(8)
print(f'alltoall之前：{data} at rank {get_rank()}')
net = AlltoAll(group_size)
data = net(data)
print(f'alltoall之后：{data} at rank {get_rank()}')
