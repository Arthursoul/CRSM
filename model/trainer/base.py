import abc
import torch
import os.path as osp

from model.utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
from model.logger import Logger


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args, osp.join(args.save_path))

        # 记录当前的 train_step 与 train_epoch
        self.train_step = 0
        self.train_epoch = 0

        # 记录最大的可能阶段数
        self.max_steps = args.episodes_per_epoch * args.max_epoch

        # 一帮子用于记录时间的东西
        self.dt, self.ft = Averager(), Averager()    # dt 表示 开始训练到数据加载完成的时间  ft 表示 正向传播一个batch需要时间
        self.bt, self.ot = Averager(), Averager()    # bt 表示 完成一次反向传播的时间   ot 表示 优化参数的时间
        self.timer = Timer()

        # 训练时候的参数变化情况
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0

    # 我们通过一些装饰器或者特殊的方法来把类里的方法虚化，虚化后的方法不能通过当前类调用，必须使用子类继承并且实现该方法才能调用该方法
    # 好家伙，python里面竟然也有虚方法，模块的名字还这么奇怪，离谱
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass

    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass

    @abc.abstractmethod
    def final_record(self):
        pass

    # 打印经过当前 epoch 学习后的 情况 保存相关内容
    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            vl, va, vap = self.evaluate(self.val_loader)    # 这里会计算在 validation 集上面的情况
            self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
            self.logger.add_scalar('val_acc', float(va), self.train_epoch)
            print('epoch {}, val, loss={:.4f} acc={:.4f}+{:.4f}'.format(epoch, vl, va, vap))

            if va >= self.trlog['max_acc']:
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model('max_acc')

    def try_logging(self, tl1, tl2, ta, tg=None):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          tl1.item(), tl2.item(), ta.item(),
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train_loss', tl2.item(), self.train_step)
            self.logger.add_scalar('train_acc', ta.item(), self.train_step)
            if tg is not None:
                self.logger.add_scalar('grad_norm', tg.item(), self.train_step)
            print('data_timer: {:.2f} sec, ' \
                  'forward_timer: {:.2f} sec,' \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                self.dt.item(), self.ft.item(),
                self.bt.item(), self.ot.item())
            )
            self.logger.dump()

    # 保存模型参数
    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
