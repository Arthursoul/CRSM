import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    Averager, count_acc,
    compute_confidence_interval,
)

from torch.cuda.amp import autocast as autocast
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)   # 这个 FSLTrainer的父类 主要用于 记录训练情况

        # 获得数据
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        # 获得 模型  无multi-gpu的情况下 model = para_model
        self.model, self.para_model = prepare_model(args)
        # 获得优化器 有optimizer 也有
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        # label 是[0,1,2,3,4] 重复 15 次   label_aux 是[0,1,2,3,4] 重复 16 次
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)  # [1,75]
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)  # [1,80]
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux

    def make_nk_label(self,n, k, ep_per_batch=1):
        label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
        label = label.repeat(ep_per_batch)
        return label
    
    def train(self):
        args = self.args
        self.model.train()  # 设置 training mode = true  启用 BatchNormalization 和 Dropout
        if self.args.fix_BN:
            self.model.encoder.eval()    # 设置 不启用 BatchNormalization 和 Dropout 的参数调整

        # 将 label 替换为 meta-baseline label
        label = self.make_nk_label(args.way, args.query, args.batch).cuda()

        # 会经过 200 个 epoch
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1

            # 这里由于在后面 valdation的时候 进行了 model.eval() 所以必须每次都重新 model.train()
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()   # training loss for all epoch
            tl2 = Averager()   # training loss for current epoch
            ta = Averager()    # training accuracy for current epoch

            start_tm = time.time()  # 记录 当前 epoch 的开始时间

            train_gen = tqdm(self.train_loader)
            # 开始 运用batch 进行训练 一个 batch 16个样本 包括了 1个support 15个query
            for i, batch in enumerate(train_gen, 1):
                self.train_step += 1

                # 将batch 中所有 样本 转移到 cuda上面
                # data => [80,3,84,84] gt_label => [1,80]
                if torch.cuda.is_available():
                    data, que, gt_label = [_.cuda() for _ in batch]
                else:
                    data, que, gt_label = batch[0], batch[1], batch[2]

                # 记录 数据准备完成的时间
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                with autocast():
                    logits = self.para_model(data, que)
                    loss = F.cross_entropy(logits, label)
                    
                tl2.add(loss)

                # 记录正向传播的时间
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)

                # 准确率
                acc = count_acc(logits, label)

                # 记录准确率
                tl1.add(loss.item())
                ta.add(acc)

                train_gen.set_description(
                    '训练阶段:epo {} total_loss={:.4f} partial_loss={:.4f} 平均acc={:.4f}'.format(epoch, tl1.item(), tl2.item(), ta.item()))

                self.optimizer.zero_grad()
                loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)
                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)
                start_tm = time.time()

            # 调整 优化器 学习率
            self.lr_scheduler.step()
            # logger 方面的记录 对训练整体过程并没有帮助
            self.try_evaluate(epoch)
            # 打印当前总共计算的时间 与 预计的总时间
            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    # 评估当前轮次的validation计算情况
    def evaluate(self, data_loader):
        args = self.args
        self.model.eval()

        record = np.zeros((args.num_eval_episodes, 2))
        label = self.make_nk_label(args.way, args.query, args.batch).cuda()

        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            val_gen = tqdm(data_loader)

            tl1 = Averager()
            ta = Averager()

            for i, batch in enumerate(val_gen, 1):
                if torch.cuda.is_available():
                    data, que, _ = [_.cuda() for _ in batch]
                else:
                    data, que = batch[0], batch[1]
                
                logits = self.model(data, que)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                
                tl1.add(loss)
                ta.add(acc)

                val_gen.set_description('验证阶段:平均loss1={:.4f} 平均acc={:.4f}'.format(tl1.item(), ta.item()))
                
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        # 切换为 train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    # 评估最后的 test 计算情况
    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((int(10000/self.args.batch), 2)) # loss and acc
        label = self.make_nk_label(args.way, args.query, args.batch).cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            tl1 = Averager()
            ta = Averager()
            test_gen = tqdm(self.test_loader)

            for i, batch in enumerate(test_gen, 1):
                if torch.cuda.is_available():
                    data, que, _ = [_.cuda() for _ in batch]
                else:
                    data, que = batch[0], batch[1]
                # with torch.enable_grad():
                logits = self.model(data, que)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc

                tl1.add(loss)
                ta.add(acc)

                test_gen.set_description('测试阶段:平均loss1={:.4f} 平均acc={:.4f}'.format(tl1.item(), ta.item()))

        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        torch.save(self.model.state_dict(), args.save_path + '_{}.pth'.format(va))

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        return vl, va, vap
    
    def final_record(self):
        # save the best performance in a txt file
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))            