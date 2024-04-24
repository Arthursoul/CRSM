import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler_meta, CategoriesSampler_metaVQA

from model.models.protonet import ProtoNet

# no usage
class MultiGPUDataloader:
    def __init__(self, dataloader, num_device):
        self.dataloader = dataloader
        self.num_device = num_device

    def __len__(self):
        return len(self.dataloader) // self.num_device

    def __iter__(self):
        data_iter = iter(self.dataloader)
        done = False

        while not done:
            try:
                output_batch = ([], [])
                for _ in range(self.num_device):
                    batch = next(data_iter)
                    for i, v in enumerate(batch):
                        output_batch[i].append(v[None])

                yield (torch.cat(_, dim=0) for _ in output_batch)
            except StopIteration:
                done = True
        return


# 根据不同的数据集 返回不同的数据加载器
def get_dataloader(args):
    # if args.dataset == 'MiniImageNet':
    #     # Handle MiniImageNet
    #     from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    # elif args.dataset == 'CUB':
    #     from model.dataloader.cub import CUB as Dataset
    # elif args.dataset == 'TieredImageNet':
    #     from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    # else:
    #     raise ValueError('Non-supported Dataset.')
    from model.dataloader.fsl_vqa import FSLVQA as Dataset

    num_device = torch.cuda.device_count()  # 这里是获取 gpu的个数，如果想要用cpu 就直接 将num_device变成0
    num_episodes = args.episodes_per_epoch * num_device if args.multi_gpu else args.episodes_per_epoch # 如果有多个GPU则可以减少episodes
    num_workers = 8

    trainset = Dataset('train', args, augment=args.augment)
    valset = Dataset('val', args, token_to_ix=trainset.token_to_ix)
    testset = Dataset('test', args, token_to_ix=trainset.token_to_ix)

    # trainset = Dataset('train', args, augment=args.augment, use_fapit=True)
    # valset = Dataset('test', args, token_to_ix=trainset.token_to_ix, use_fapit=True)
    # testset = Dataset('test', args, token_to_ix=trainset.token_to_ix, use_fapit=True)
    args.num_class = trainset.num_class  # 样本类别数

    train_sampler = CategoriesSampler_metaVQA(trainset.label2ind,
                                           num_episodes,
                                           max(args.way, args.num_classes),
                                           args.shot + args.query + args.unlabeled, args.batch)
    train_loader = DataLoader(dataset=trainset,
                              num_workers=num_workers,
                              batch_sampler=train_sampler,
                              pin_memory=True)
    val_sampler = CategoriesSampler_metaVQA(valset.label2ind,
                                         args.num_eval_episodes,
                                         args.eval_way, args.eval_shot + args.eval_query + args.eval_unlabeled, args.batch)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    test_sampler = CategoriesSampler_metaVQA(testset.label2ind,
                                          int(10000 / args.batch),  # args.num_eval_episodes,
                                          args.eval_way, args.eval_shot + args.eval_query + args.eval_unlabeled, args.batch)
    test_loader = DataLoader(dataset=testset,
                             batch_sampler=test_sampler,
                             num_workers=args.num_workers,
                             pin_memory=True)
    
    args.pretrained_emb = trainset.pretrained_emb
    args.token_size = trainset.token_size

    return train_loader, val_loader, test_loader


# 加载 embedding adaptation 模型
def prepare_model(args):
    # 通过字符串的方式  eval(“string”) 执行string 代表的表达式
    model = eval(args.model_class)(args)

    # load pre-trained model (no FC weights)
    

    if torch.cuda.is_available():
        # 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
        # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
        # 反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        para_model = model.to(device)
    else:
        para_model = model.to(device)

    return model, para_model


# 设置不同的参数优化器
def prepare_optimizer(model, args):
    # model.named_parameters()  给出网络层的名字和参数的迭代器  这里将所有的 没有 encoder 的parameter取了出来
    # 但是有的时候就是空的
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]

    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    if args.backbone_class == 'ConvNet':
        optimizer = optim.Adam(
            [{'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )
    else:
        optimizer = optim.SGD(
            [{'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )

    # torch.optim.lr_scheduler模块提供了一些根据epoch训练次数来调整学习率（learning rate）的方法。
    # 一般情况下我们会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果。
    # 而torch.optim.lr_scheduler.ReduceLROnPlateau则提供了基于训练中某些测量值使学习率动态下降的方法。
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(_) for _ in args.step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.max_epoch,
            eta_min=0  # a tuning parameter
        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
