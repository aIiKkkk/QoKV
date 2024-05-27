import torch
import torch.nn as nn
import logging
logger = logging.getLogger('OS+')
scale_list = []

def migration(K, Q, out_projweight, num_heads, head_dim): #, a_qconfig, w_qconfig, module_type):
    migrator = Migrator1DRangeSearch(K, Q, out_projweight, num_heads, head_dim) #, a_qconfig, w_qconfig, module_type)
    best_scale = migrator()
    scale_list.append(best_scale)
    return best_scale

class MigratorBase(nn.Module):
    def __init__(self, input, weight, out_projweight, num_heads, head_dim): #, a_qconfig, w_qconfig, module_type, extra_dict=None):
        super().__init__()
        self.inputdim = weight.shape[0]
        if self.inputdim == num_heads:
            self.input = input.view(-1, weight.shape[1], num_heads * head_dim)
            self.weight = weight
        else:
            self.input = input
            self.weight = weight
        self.out_projweight = out_projweight
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = self.input.dtype
        self.device = self.input.device
        # calculate min max in advance
        self.cmx = self.input.max(0)[0].max(0)[0]  # 计算整个张量的最大最小值  # 可以改成计算行维度或列维度的最大最小值
        self.cmn = self.input.min(0)[0].min(0)[0]
        self.amx = max(self.input.max(), torch.tensor(0.0, dtype=self.dtype).to(self.device))
        self.amn = min(self.input.min(), torch.tensor(0.0, dtype=self.dtype).to(self.device))
        logger.info('the data type is {}, the device is {}'.format(self.dtype, self.device))
        logger.info('the activation range is {:.2f}, {:.2f}'.format(self.amn, self.amx))
        logger.info('the weighange is {:.2f}, {:.2f}'.format(self.weight.min(), self.weight.max()))
        # calculate output
        self.output = self.get_output(self.input, self.weight, self.out_projweight )
        # prepare MinMax Observer for later Quantizet
    def get_output(self, input, weight, qout_projweight):
        # proj_shape = (self.num_heads, -1, self.head_dim)
        # output = torch.bmm(weight.view(proj_shape), input.view(proj_shape).transpose(1, 2))
        if self.inputdim == self.num_heads:
            input = input.view(self.num_heads, weight.shape[1], self.head_dim)
            output = torch.bmm(weight, input)
            output = output.view(-1, weight.shape[1], self.num_heads * self.head_dim)
            output = torch.bmm(output, qout_projweight.transpose(0, 1).unsqueeze(0))
        else:
            output = torch.bmm(weight, input.transpose(1, 2))
        return output

    def quantize(self, X, n_bits=8):
        # print(X.shape)
        xmax = X.abs().amax()
        q_max = 2 ** (n_bits - 1) - 1
        scales = xmax / q_max
        scales.clamp_(min=1e-5)
        s = (X / scales).round_()
        s = s.mul_(scales)
        return s

    def get_qoutput(self, input, weight, out_projweight, cur_scale, clipping_range=None):
        if self.inputdim == self.num_heads:
            qinput = self.quantize(input / cur_scale, n_bits=4) * cur_scale
            qweight = weight
            qout_projweight = self.quantize(out_projweight * cur_scale, n_bits=8)
        else:
            qinput = self.quantize(input / cur_scale, n_bits=8)
            qweight = self.quantize(weight * cur_scale , n_bits=8)
        return self.get_output(qinput, qweight, qout_projweight)

    def cac_scale(self, min_range, max_range):
        mx_scale = torch.where(self.cmx > max_range, self.cmx / max_range, torch.tensor(1.0, dtype=self.dtype).to(self.device))
        mn_scale = torch.where(self.cmn < min_range, self.cmn / min_range, torch.tensor(1.0, dtype=self.dtype).to(self.device))
        final_scale = torch.max(mx_scale, mn_scale)
        return final_scale

    def get_best_scale(self, min_range, max_range):
        best_scale = self.cac_scale(min_range, max_range)
        return best_scale

    def loss_fx(self, pred, tgt, p=2.0):
        return (pred - tgt).abs().pow(p).sum(-1).mean()

    def cac_loss(self, min_range, max_range):
        cur_scale = self.cac_scale(min_range, max_range)  # 计算缩放值
        qoutput = self.get_qoutput(self.input , self.weight, self.out_projweight, cur_scale, (min_range, max_range))
        return self.loss_fx(qoutput, self.output)

    def forward(self,):
        pass


class Migrator1DRangeSearch(MigratorBase):
    # if adopting it, we shall first shift the values
    def __init__(self, K, Q, out_projweight, num_heads, head_dim): #, a_qconfig, w_qconfig, module_type, extra_dict=None):
        super().__init__(K, Q, out_projweight, num_heads, head_dim)  #, a_qconfig, w_qconfig, module_type, extra_dict)
        self.num = max(100, int(self.amx / 0.5))

    def cac_scale_loss(self, mn_range, mx_range):
        return self.cac_loss(torch.tensor(mn_range, dtype=self.dtype).to(self.device),
                             torch.tensor(mx_range, dtype=self.dtype).to(self.device))
    # 把最大最小范围转化为合适的格式，并放到指定的设备

    def search_migrate_range_1D(self,):
        best_loss = None
        bounds = (1.0, max(-self.amn.item(), self.amx.item()))  # 取输入张量的最大绝对值
        step = (bounds[1] - bounds[0]) / self.num  # 确定每一步的步长
        mn_range = -bounds[1]
        mx_range = bounds[1]
        st = bounds[1]
        cnt = 0
        while st >= bounds[0]:
            loss = self.cac_scale_loss(-st, st)
            if best_loss is None or best_loss > loss:
                best_loss = loss
                mn_range = -st
                mx_range = st
            cnt += 1
            if cnt % 10 == 0:
                logger.info('{:.2f} loss at iter {}'.format(loss, cnt))
            st -= step

        return (torch.tensor(mn_range, dtype=self.dtype).to(self.device),
                torch.tensor(mx_range, dtype=self.dtype).to(self.device))

    def forward(self,):
        best_range = self.search_migrate_range_1D()
        return self.get_best_scale(*best_range)