import torch
import torch.nn as nn


DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():  # 迭代当前模块的子模块，每次迭代产生子模块的名称和子模块本身
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))  # 递归调用 find_layers 函数来查找子模块中的目标层，并将结果更新到字典 res 中
    return res
