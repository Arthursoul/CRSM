import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric
from model.networks.backbone import VQABackbone
import torch.nn.functional as F
    
class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
   
            
        self.encoder = VQABackbone(args, pre_train=True)                      


    def forward(self, img, que):
        out = self.encoder(img, que)
        return out
    
    def forward_proto(self, data_shot, data_query, que_shot, que_query, way):
        proto = self.encoder(data_shot, que_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        query = self.encoder(data_query, que_query)
        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim