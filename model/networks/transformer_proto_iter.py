

import torch.nn as nn
import torch.nn.functional as F
import torch, math

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return - (x - y)**2 + F.relu(x + y)
    

def _move_ptr_fw(stack_ptr):
    filter_fw = torch.FloatTensor([1, 0, 0]).view(1, 1, 3).to(stack_ptr.device)
    batch_size, stack_len = stack_ptr.size()
    new_stack_ptr = F.conv1d(stack_ptr.view(batch_size, 1, stack_len), filter_fw, padding=1).view(batch_size, stack_len)
    stack_top_mask = torch.zeros(stack_len).to(stack_ptr.device)
    stack_top_mask[stack_len - 1] = 1 # [stack_len, ]
    new_stack_ptr += stack_top_mask * stack_ptr
    return new_stack_ptr


def _write_to_stack(att_stack, stack_ptr, att):
    batch_size, stack_len = stack_ptr.size()
    stack_ptr_expand = stack_ptr.view(batch_size, 1, 1, stack_len)
    if att.dim() == 3:
        att = att.unsqueeze(3)
    att_stack = att * stack_ptr_expand + att_stack * (1 - stack_ptr_expand)
    return att_stack # (batch_size, att_dim, glimpse, stack_len)


class MHAtt(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(MHAtt, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / 8)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q = nn.Linear(hidden_dim,hidden_dim)
        self.linear_merge = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q, mask=None):
        b, n, s = q.shape

        v = self.linear_v(v).view(b, -1, self.head, self.head_size).transpose(1, 2)
        k = self.linear_k(k).view(b, -1, self.head, self.head_size).transpose(1, 2)
        q = self.linear_q(q).view(b, -1, self.head, self.head_size).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(b, -1, self.hidden_dim)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -65504.0)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, outdim=640):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_r)
        self.dense2 = nn.Linear(hidden_dim * 2, outdim)

    def forward(self, X):
        return self.dense2(self.dropout(self.relu(self.dense1(X))))

class AttFlat(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, out_dim=640, glimpses=1):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses

        self.mlp = PositionWiseFFN(hidden_dim, dropout_r, self.glimpses)

        self.linear_merge = nn.Linear(
            hidden_dim * glimpses,
            out_dim
        )

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -65504.0
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class Encoder(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(Encoder, self).__init__()

        self.mhatt = MHAtt(hidden_dim, dropout_r, head)
        self.ffn = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x
    

# class Decoder(nn.Module):
#     def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
#         super(Decoder, self).__init__()

#         self.mhatt1 = MHAtt(hidden_dim, dropout_r, head)
#         self.mhatt2 = MHAtt(hidden_dim, dropout_r, head)
#         self.ffn = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)

#         self.dropout1 = nn.Dropout(dropout_r)
#         self.norm1 = nn.LayerNorm(hidden_dim)

#         self.dropout2 = nn.Dropout(dropout_r)
#         self.norm2 = nn.LayerNorm(hidden_dim)

#         self.dropout3 = nn.Dropout(dropout_r)
#         self.norm3 = nn.LayerNorm(hidden_dim)

#     def forward(self, x, y, x_mask, y_mask):
#         x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))

#         x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
#         x = self.norm3(x + self.dropout3(self.ffn(x)))

#         return x


class FindModule(nn.Module):

    def __init__(self, hidden_dim, dropout_r, glimpses=2):
        super().__init__()
        concept_weight = torch.load('./prototype/prototype_vqa.pt') # glove prototypes
        self.num_concept, self.concept_dim = concept_weight.shape 
        self.mem2concept = nn.Linear(hidden_dim,self.concept_dim) # dim_vision x dim
        self.prototype_layer = nn.Linear(self.concept_dim,self.num_concept, bias=False)
        self.prototype_layer.weight.data = concept_weight
        for para in self.prototype_layer.parameters():
            para.requires_grad = False # originally False

        self.map_c = nn.Linear(hidden_dim, self.num_concept)
        self.x_conv = nn.Linear(self.num_concept, glimpses)
        self.drop = nn.Dropout(dropout_r)
        self.fusion = Fusion()

        self.sem2hidden = nn.Linear(self.num_concept*glimpses,hidden_dim)


    def forward(self, feat, query):
        b, n, h = feat.shape
        query = torch.tanh(self.map_c(self.drop(query))) # (batch_size, 1, dim_v)

        # prototype
        mem_concept = self.mem2concept(feat)
        feat = torch.tanh(self.prototype_layer(mem_concept)) # (batch_size, num_feat, num_concept)

        x = self.fusion(feat, query)
        att_out = self.x_conv(self.drop(x)) # (batch_size, num_feat, glimpse)
        att_out = F.softmax(att_out, dim=1)  # (batch_size, num_feat, glimpse)
        


        att_sem = torch.bmm(att_out.transpose(2,1),feat).view(len(feat),-1)
        semantic = self.sem2hidden(att_sem)

        return semantic.unsqueeze(1).expand(b, n, h)


class Decoder(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(Decoder, self).__init__()

        self.mhatt1 = MHAtt(hidden_dim, dropout_r, head)
        self.mhatt2 = MHAtt(hidden_dim, dropout_r, head)
        self.ffn = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout3 = nn.Dropout(dropout_r)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.find = FindModule(hidden_dim, dropout_r)
        
        self.dropout4 = nn.Dropout(dropout_r)
        self.norm4 = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, x_mask, y_mask, query):
        query = sum(query).permute(1, 0, 2) / 2;
        find_semantic = self.find(x, query)
        x = self.norm4(x + self.dropout4(find_semantic))

        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))

        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))

        return x


class Transformer(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8, avg_pool=True):
        super(Transformer, self).__init__()
        self.avg_pool = avg_pool

        self.enc_list = nn.ModuleList([Encoder(hidden_dim, dropout_r, head) for _ in range(6)])
        self.dec_list = nn.ModuleList([Decoder(hidden_dim, dropout_r, head) for _ in range(6)])
        
        

        if avg_pool:
            self.img_avgpool = nn.AdaptiveAvgPool1d(1)
            self.que_avgpool = nn.AdaptiveAvgPool1d(1)
            # self.que_flatten = AttFlat(hidden_dim, dropout_r, hidden_dim)
            # self.img_flatten = AttFlat(hidden_dim, dropout_r, hidden_dim)
        

    def forward(self, img, que, img_mask, que_mask, query):

        for enc in self.enc_list:
            que = enc(que, que_mask)
        
        
        b, n, c = img.shape
        for dec in self.dec_list:
            img = dec(img, que, img_mask, que_mask, query)

        if self.avg_pool:
            img = self.img_avgpool(img.permute(0, 2, 1)).view(b, -1)
            que = self.que_avgpool(que.permute(0, 2, 1)).view(b, -1)
            # img = self.img_flatten(img, img_mask)
            # que = self.que_flatten(que, que_mask)

        return img, que

