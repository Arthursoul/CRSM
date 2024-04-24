import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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

class Prototype_Module(nn.Module):
    def __init__(self,img_size,num_prototype,num_concept, token_size, pretrained_emb, dropout_r=0.1):
        super(Prototype_Module,self).__init__()
        self.num_prototype = num_prototype
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=300
            )
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        # raw features
        self.hidden_size = img_size	
        self.label = nn.Linear(300, 512,bias=False)
        self.fuse = nn.Linear(img_size + 512, img_size,bias=False)
        self.prototype = nn.Linear(img_size, num_prototype,bias=False)
        	
        self.mhatt = MHAtt(self.hidden_size)
        self.dropout = nn.Dropout(dropout_r)
        self.norm = nn.LayerNorm(self.hidden_size)

        self.mhatt_label = MHAtt(512)
        self.dropout_label = nn.Dropout(dropout_r)
        self.norm_label = nn.LayerNorm(512)

        self.proto2concept = nn.Linear(self.hidden_size,num_concept) 

        self.attention_layer = nn.Linear(self.hidden_size,1)
        

    def forward(self,img, label_ix):
        batch = len(img)
        label = self.label(self.embedding(label_ix))

        label = self.norm_label(label + self.dropout_label(self.mhatt_label(label, label, label)))

        multi = self.fuse(torch.cat((img, label), dim=-1))
        proto_sim = torch.sigmoid(self.prototype(multi)) # originally tanh
        merged_proto = torch.bmm(proto_sim,self.prototype.weight.unsqueeze(0).expand(batch,self.num_prototype,self.hidden_size))
        multi = self.norm(multi + self.dropout(self.mhatt(multi, multi, multi)))
        proto_sim_later = torch.sigmoid(self.prototype(multi)) # originally tanh
        merged_proto_later = torch.bmm(proto_sim_later,self.prototype.weight.unsqueeze(0).expand(batch,self.num_prototype,self.hidden_size))
        # attentive prediction

        prediction = torch.sigmoid(self.proto2concept(merged_proto + merged_proto_later))
        att = F.softmax(self.attention_layer(merged_proto),dim=1)
        prediction = (prediction*att).sum(1)

        return prediction
