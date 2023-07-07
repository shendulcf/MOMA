import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from topk import SmoothTop1SVM

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) 
        # 一个线性层，将输入的dim维度映射到inner_dim * 3维度，即将输入映射为查询（query）、键（key）和值（value）。
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) 

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # Q*K/d^-0.5
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

'''
在Transformer类的初始化函数中，定义了模型的结构。它接受以下参数：

dim：表示Transformer模型的隐藏维度。
depth：表示Transformer模型的层数，即堆叠的Transformer层的数量。
heads：表示自注意力机制中的注意力头数。
dim_head：表示每个注意力头的维度。
mlp_dim：表示前馈神经网络的隐藏层维度。
dropout：表示用于正则化的丢弃概率。
在初始化函数中，使用一个循环来创建多个Transformer层，并将它们添加到模型的layers列表中。每个Transformer层由两个部分组成：

Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)))：表示一个具有残差连接的预标准化（PreNorm）的自注意力机制。它将输入通过层归一化（LayerNorm）处理后，传递给自注意力机制，然后将自注意力机制的输出与输入进行残差连接。

Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))：表示一个具有残差连接的预标准化的前馈神经网络。它将输入通过层归一化处理后，传递给前馈神经网络，然后将前馈神经网络的输出与输入进行残差连接。

这样，通过堆叠多个Transformer层，构建了一个完整的Transformer模型。

在forward函数中，通过迭代遍历模型的所有Transformer层，并依次对输入进行自注意力机制和前馈神经网络的计算。最后返回输出。
'''
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class DualPathTransformer(nn.Module):
    def __init__(self, input_feats_dim, depth, num_heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers_low = nn.ModuleList([])
        self.layers_high = nn.ModuleList([])
        for _ in range(depth):
            self.layers_low.append(nn.ModuleList([
                Residual(PreNorm(input_feats_dim, Attention(input_feats_dim, heads = num_heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(input_feats_dim, FeedForward(input_feats_dim, mlp_dim, dropout = dropout)))
            ]))
            self.layers_high.append(nn.ModuleList([
                Residual(PreNorm(input_feats_dim, Attention(input_feats_dim, heads = num_heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(input_feats_dim, FeedForward(input_feats_dim, mlp_dim, dropout = dropout)))
            ]))
        # self.fusion_linear = nn.Linear(input_feats_dim * 2, mlp_dim)
    def forward(self, x1, x2, mask=None):
        for attn, ff in self.layers_low:
            x1 = attn(x1, mask = mask)
            x1 = ff(x1)
        for attn, ff in self.layers_high:
            x2 = attn(x2, mask = mask)
            x2 = ff(x2)
        print(x1.size())
        print(x2.size())
        x = torch.cat((x1, x2), dim=1)
        
        # x = self.fusion_linear(x)
        return x
'''
Residual类和PreNorm类是用于模型中的残差连接和层归一化的辅助类。
FeedForward类定义了一个前馈神经网络层，用于对输入进行非线性变换。
Attention类定义了注意力机制，包括将输入映射为查询（query）、键（key）和值（value），计算注意力权重，并将注意力应用于值以得到输出。
Transformer类定义了一个多层的Transformer模型，由多个注意力层和前馈神经网络层组成。
MIL类是多示例学习模型的主类。它包含了注意力计算、实例级别评估和总体损失计算的功能。
这是多示例学习模型的主类。在初始化函数中，定义了模型的结构，包括注意力机制、分类器等组件。

inst_eval函数用于对每个示例进行评估，根据输入的注意力权重计算实例损失，并返回预测结果和目标标签。

inst_eval_out函数用于对不在类别中的示例进行评估，计算实例损失，并返回预测结果和目标标签。

forward函数定义了模型的前向传播过程，包括注意力计算、实例级别评估和总体损失计算。

最后的SmoothTop1SVM是另一个模块，可能是用于执行SmoothTop1SVM算法的。在提供的代码中没有给出相关的具体实现，所以无法给出更多详细信息。
'''


class MIL(nn.Module):
    def __init__(self, hidden_dim = 512, num_class = 2, encoder_layer = 1, k_sample = 2, tau = 0.7):
        super().__init__()
        
        self.k_sample = k_sample
        self.n_classes = num_class
        self.L = hidden_dim
        self.D = hidden_dim
        self.K = 1
        self.subtyping = True
        
        self.instance_loss_fn = SmoothTop1SVM(num_class, tau = tau).cuda()
        # 这是用于计算注意力权重的两个线性层，输入是隐藏层表示H，输出经过激活函数处理后得到注意力的两个部分
        self.attention_V2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights2 = nn.Linear(self.D, self.K) # 该线性层用于将注意力的两个部分相乘并降维为一个值，用于后续的Softmax操作

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 2)
        )

        instance_classifiers = [nn.Linear(hidden_dim, 2) for i in range(num_class)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers) # 一个多个线性层的列表，用于对多个类的示例进行分类
        
        self.cls_token = nn.Parameter(torch.zeros((1, 1, hidden_dim))) # 一个可学习的参数，用于表示每个输入序列的类别
        self.projector = nn.Linear(2048, hidden_dim) # 一个线性层，用于将输入序列的维度从2048投影到隐藏层维度
        self.transformer = Transformer(hidden_dim, encoder_layer, 8, 64, 2048, 0.1)
        self.dropout = nn.Dropout(0.1)
        
        
        
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device, dtype = torch.long)
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device, dtype = torch.long)
    
    def inst_eval(self, A, h, instance_feature, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        #top_p = torch.index_select(instance_feature, dim=0, index=top_p_ids)
        top_p = [instance_feature[i] for i in top_p_ids]
        top_p = torch.cat(top_p, dim = 1).squeeze(0) 
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        #top_n = torch.index_select(instance_feature, dim=0, index=top_n_ids)
        top_n = [instance_feature[i] for i in top_n_ids]
        top_n = torch.cat(top_n, dim = 1).squeeze(0)
        p_targets = self.create_positive_targets(len(top_p), device)
        n_targets = self.create_negative_targets(len(top_n), device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, instance_feature, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, len(A))[1][-1]
        top_p = [instance_feature[i] for i in top_p_ids]
        top_p = torch.cat(top_p, dim = 1).squeeze(0)
        p_targets = self.create_negative_targets(len(top_p), device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets    
    
    def forward(self, xs, label):
        H = []
        instance_feature = []
        for x in xs:
            #x = x.permute(1,0, 2)
            x = self.projector(x)
            x = torch.cat((self.cls_token, x), dim = 1)
            x = self.dropout(x)
            rep = self.transformer(x) # b,n,(h,d) --> b,n,dim --> b,n, mlp_dim
            H.append(rep[:, 0]) # class_token
            instance_feature.append(rep[:, 1:])
            
        H = torch.cat(H)
        A_V = self.attention_V2(H)  # NxD
        A_U = self.attention_U2(H)  # NxD
        A = self.attention_weights2(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        
        
        total_inst_loss = 0.0
        all_preds = []
        all_targets = []
        inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
        for i in range(len(self.instance_classifiers)):
            inst_label = inst_labels[i].item() # 用于将张量或张量中的元素转换为Python标量
            classifier = self.instance_classifiers[i]
            if inst_label == 1: #in-the-class:
                instance_loss, preds, targets = self.inst_eval(A, H, instance_feature, classifier)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            else: #out-of-the-class
                if self.subtyping:
                    instance_loss, preds, targets = self.inst_eval_out(A, H, instance_feature, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    continue
            total_inst_loss += instance_loss        
        
        
        total_inst_loss /= len(self.instance_classifiers)
        
        
        
        M = torch.mm(A, H)  # KxL        
        logit = self.classifier(M)
    
        
        return logit, total_inst_loss, A.detach()

if __name__ == "__main__":
    input1 = torch.randn(1, 10, 512)
    input2 = torch.randn(1, 32, 512)
    model = DualPathTransformer(512, 1, 8, 64, 2048, 0.1)
    output = model(input1, input2)

    print(model)
    print(output.size())