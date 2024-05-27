import torch
import torch.nn as nn
import logging
logger = logging.getLogger('OS+')
scale_list = []

def migrationk(key_states, query_states, Kweight , Kbias,  Qweight , Qbias, hidden_states, num_heads, head_dim):
    migrator = Migrator1DRangeSearch(key_states, query_states, Kweight , Kbias,  Qweight , Qbias, hidden_states, num_heads, head_dim)
    best_scale = migrator()
    scale_list.append(best_scale)
    return best_scale

class MigratorBase(nn.Module):
    def __init__(self, key_states, query_states, Kweight , Kbias, Qweight , Qbias, hidden_states, num_heads, head_dim):
        super().__init__()
        self.input = key_states
        self.query_states = query_states
        self.Kweight = Kweight
        self.Kbias = Kbias
        self.Qweight = Qweight
        self.Qbias = Qbias
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = self.input.dtype
        self.device = self.input.device
        # calculate min max in advance
        self.cmx = self.input.max(0)[0].max(0)[0]  # 计算整个张量的最大最小值  # 可以改成计算行维度或列维度的最大最小值
        self.cmn = self.input.min(0)[0].min(0)[0]
        self.amx = max(self.input.max(), torch.tensor(0.0, dtype=self.dtype).to(self.device))
        self.amn = min(self.input.min(), torch.tensor(0.0, dtype=self.dtype).to(self.device))
        # calculate output
        self.output = self.get_output(self.Kweight, self.Kbias, self.Qweight, self.Qbias, self.hidden_states)
        # prepare MinMax Observer for later Quantize

    def get_output(self, Kweight, Kbias, Qweight, Qbias, hidden_states):
        bsz, tgt_len, _ = hidden_states.size()
        Q = (torch.bmm(hidden_states, Qweight.transpose(0, 1).unsqueeze(0))) + Qbias
        # Q = Q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # Q = Q.view((bsz * self.num_heads, -1, self.head_dim))
        K = (torch.bmm(hidden_states, Kweight.transpose(0, 1).unsqueeze(0))) + Kbias
        # K = K.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # K = K.view((bsz * self.num_heads, -1, self.head_dim))
        output = torch.bmm(Q, K.transpose(1, 2))
        return output

    def quantize(self, X, axis=-1, n_bits=4):
        xmax = X.abs().amax(axis, keepdim=True)[0]
        q_max = 2 ** (n_bits - 1) - 1
        scales = xmax / q_max
        scales.clamp_(min=1e-5)
        s = (X / scales).round_()
        s = s * (scales)
        return s

    def quantize_w(self, X, n_bits=8):
        xmax = X.abs().amax()
        q_max = 2 ** (n_bits - 1) - 1
        scales = xmax / q_max
        scales.clamp_(min=1e-5)
        s = (X / scales).round_()
        s = s * (scales)
        return s

    def get_qoutput(self, Kweight, Kbias, Qweight, Qbias, hidden_states ,cur_scale, clipping_range=None):
        bsz, tgt_len, _ = hidden_states.size()
        Qbias = Qbias * (cur_scale)
        cur_scale = cur_scale.unsqueeze(0)
        cur_scale_T = cur_scale.transpose(-2, -1)
        Qweight = Qweight * (cur_scale_T)
        qQweight = self.quantize_w(Qweight, n_bits=8)
        Q = (torch.bmm(hidden_states, qQweight.transpose(0, 1).unsqueeze(0))) + Qbias
        Q = self.quantize(Q, n_bits=8)
        # Q = Q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # Q = Q.view((bsz * self.num_heads, -1, self.head_dim))

        Kbias = Kbias / (cur_scale)
        Kweight = Kweight / (cur_scale_T)
        qKweight = self.quantize_w(Kweight, n_bits=8)
        K = (torch.bmm(hidden_states, qKweight.transpose(0, 1).unsqueeze(0))) + Kbias
        K = self.quantize(K, n_bits=8)
        # K = K.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # K = K.view((bsz * self.num_heads, -1, self.head_dim))

        output = torch.bmm(Q, K.transpose(1, 2))
        return output


    def cac_scale(self, min_range, max_range):
        mx_scale = torch.where(self.cmx > max_range, self.cmx / max_range, torch.tensor(1.0, dtype=self.dtype).to(self.device))
        # 可以改成计算行或列维度的缩放值
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
        qoutput = self.get_qoutput(self.Kweight, self.Kbias, self.Qweight, self.Qbias, self.hidden_states, cur_scale,  (min_range, max_range))
        return self.loss_fx(qoutput, self.output)

    def forward(self,):
        pass

class Migrator1DRangeSearch(MigratorBase):
    # if adopting it, we shall first shift the values
    def __init__(self, key_states, query_states, Kweight , Kbias, Qweight , Qbias, hidden_states, num_heads, head_dim): #, a_qconfig, w_qconfig, module_type, extra_dict=None):
        super().__init__(key_states, query_states, Kweight , Kbias, Qweight , Qbias, hidden_states, num_heads, head_dim)  #, a_qconfig, w_qconfig, module_type, extra_dict)
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