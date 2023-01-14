import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore.communication import init, get_rank

init()
data = mnp.ones((1, 2)) * (get_rank() + 1)
print(f'allreduce之前：{data} at rank {get_rank()}')
allreduce_sum = ops.AllReduce(ops.ReduceOp.SUM)
data = allreduce_sum(data)
print(f'allreduce之后：{data} at rank {get_rank()}')
