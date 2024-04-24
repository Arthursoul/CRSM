import torch
import torch.nn as nn
import numpy as np
from model.networks.lstm import LSTM
from model.networks.transformer import Transformer


def make_mask(feature):
    return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


class FewShotModel(nn.Module):
    def __init__(self, args, hidden_dim=768):
        super().__init__()
        self.args = args
        hidden_dim = 512
        self.img_linear = nn.Linear(2048, hidden_dim)
        # self.que_encoder = LSTM(args.pretrained_emb, args.token_size, hidden_dim=hidden_dim, avg_pool=False)
        self.que_encoder = LSTM(args.pretrained_emb, args.token_size, hidden_dim=hidden_dim)
        # self.transformer = Transformer(hidden_dim=hidden_dim)
        self.multi_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.img_avgpool = nn.AdaptiveAvgPool1d(1)


    # 划分 数据集 与 训练集
    # 这里返回的是 训练集 或者 测试集 index
    # 如 5-way-1-shot query: 3个    就会是 train: [[[0,1,2,3,4,]]]
    # test: [[[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19]]]
    def split_instances(self, data):
        args = self.args
        if self.training:
            return (torch.Tensor(np.arange(args.way * args.shot)).long().view(1, args.shot, args.way),
                    torch.Tensor(np.arange(args.way * args.shot, args.way * (args.shot + args.query))).long().view(1,
                                                                                                                   args.query,
                                                                                                                   args.way))
        else:
            return (
                torch.Tensor(np.arange(args.eval_way * args.eval_shot)).long().view(1, args.eval_shot, args.eval_way),
                torch.Tensor(np.arange(args.eval_way * args.eval_shot,
                                       args.eval_way * (args.eval_shot + args.eval_query))).long().view(1,
                                                                                                        args.eval_query,
                                                                                                        args.eval_way))

    def split_shot_query(self, data, que, ep_per_batch=1):
        args = self.args  #加上旋转的torch.Size([80, 4, 3, 84, 84])
        img_shape = data.shape[1:]
        data = data.view(ep_per_batch, args.way, args.shot + args.query, *img_shape)
        x_shot, x_query = data.split([args.shot, args.query], dim=2)
        x_shot = x_shot.contiguous()
        x_query = x_query.contiguous().view(ep_per_batch, args.way * args.query, *img_shape)

        que_shape = que.shape[1:]
        que = que.view(ep_per_batch, args.way, args.shot + args.query, *que_shape)
        que_shot, que_query = que.split([args.shot, args.query], dim=2)
        que_shot = que_shot.contiguous()
        que_query = que_query.contiguous().view(ep_per_batch, args.way * args.query, *que_shape)
        return x_shot, x_query, que_shot, que_query

    def forward(self, x, que, get_feature=False):
        if get_feature:
            # 我不知道为什么会有这个 get_feature 但就运行的顺序来看，这个get_feature这辈子都不可能是true呀
            return x
        else:
            x_shot, x_query, que_shot, que_query = self.split_shot_query(x, que, self.args.batch)

            shot_shape = x_shot.shape[:-2]
            query_shape = x_query.shape[:-2]
            img_shape = x_shot.shape[-2:]
            que_shape = que_shot.shape[-1:]

            x_shot = x_shot.view(-1, *img_shape)
            x_query = x_query.view(-1, *img_shape)
           
            x_tot = torch.cat([x_shot, x_query], dim=0)
            img_mask = make_mask(x_tot)
            x_tot = self.img_linear(x_tot)
            que_shot = que_shot.view(-1, *que_shape)
            que_query = que_query.view(-1, *que_shape)
            que_tot = torch.cat([que_shot, que_query], dim=0)
            que_mask = make_mask(que_tot.unsqueeze(2))
            que_tot, query = self.que_encoder(que_tot)

            # x_tot, que_tot = self.transformer(x_tot, que_tot, img_mask, que_mask)

            # multi_tot = self.multi_linear(torch.cat([x_tot, que_tot], dim=-1))
            b, n, c = x_tot.shape
            multi_tot = self.multi_linear(torch.cat([self.img_avgpool(x_tot.permute(0, 2, 1)).view(b, -1), que_tot], dim=-1))

            feat_shape = multi_tot.shape[1:]

            x_shot, x_query = multi_tot[:len(x_shot)], multi_tot[-len(x_query):]
            x_shot = x_shot.view(*shot_shape, *feat_shape)
            x_query = x_query.view(*query_shape, *feat_shape)

            logits = self._forward(x_shot, x_query)
            return logits

    def _forward(self, x_shot, x_query):
        raise NotImplementedError('Suppose to be implemented by subclass')


if __name__ == '__main__':
    pass
