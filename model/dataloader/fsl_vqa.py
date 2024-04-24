import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import en_vectors_web_lg
import os, re

IMAGE_PATH = {'VQAv2' : '/home/public/vqa/fsl/VQA_v2/vinvl',
             'COCO' : '/home/public/vqa/fsl/VQA_v2/vinvl', 
             'VG_QA' : '/home/public3/lxq/fsl_vqa/VG_QA/VG'}
SPLIT_PATH = {'VQAv2' : '/home/public/vqa/fsl/VQA_v2',
             'COCO' : '/home/public/vqa/fsl/COCO_QA/object',
             'VG_QA' : '/home/public3/lxq/fsl_vqa/VG_QA'}

def identity(x):
    return x


def split(que):
    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        que.lower()
    ).replace('-', ' ').replace('/', ' ').split()
    return words


def proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    words = split(ques)
    
    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def tokenize(total_words, use_glove=True):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)

    for word in total_words:
        if word not in token_to_ix:
            token_to_ix[word] = len(token_to_ix)
            if use_glove:
                pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds

def proc_img_feat(img_feat, img_feat_pad_size):
    if img_feat.shape[0] > img_feat_pad_size:
        img_feat = img_feat[:img_feat_pad_size]

    img_feat = np.pad(
        img_feat,
        ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_feat

class FSLVQA(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args, augment=False, max_seq=75, max_token=15, token_to_ix=None, use_fapit=False):
        img_path = IMAGE_PATH[args.dataset]
        sp_path = SPLIT_PATH[args.dataset]
        self.max_token = max_token
        self.max_seq = max_seq
        if use_fapit:
            data_path = osp.join(sp_path, setname + '_fpait.pth')
        else:
            data_path = osp.join(sp_path, setname + '.pth')
        datas = torch.load(data_path)

        self.data = []
        self.img_ids = []
        self.que = []
        self.label = []
        self.ans_set = []

        for line in tqdm(datas['data'], ncols=64):
            self.que.append(line['question'])
            self.data.append(os.path.join(img_path, line['img_path']))
            self.img_ids.append(line['img_id'])
            if line['answer'] not in self.ans_set:
                self.ans_set.append(line['answer'])
            self.label.append(self.ans_set.index(line['answer']))

        self.num_class = len(set(self.label))

        if setname == 'train':
            self.token_to_ix, self.pretrained_emb = tokenize(datas['all_words'])
        else:  # 'val' or 'test' ,
            self.token_to_ix = token_to_ix
        self.token_size = len(self.token_to_ix)
        print('Loading {} dataset -phase {}, word size {}'.format(args.dataset, setname, self.token_size))

        self.label2ind = buildLabelIndex(self.label)

        if args.backbone_class == 'Res12':
            image_size = 84
            resize = 92
        elif args.backbone_class in ['SwinT', 'VitS']:
            image_size = 224
            resize = 256
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(resize),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class in ['SwinT', 'VitS']:
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])            
              
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, que, label = self.data[i], self.que[i], self.label[i]

        img_feat = np.load(data + '.npz')
        img_feat_x = img_feat['x'].transpose((1, 0))
        
        img_feat_iter = proc_img_feat(img_feat_x, self.max_seq)
        
        que = proc_ques(que, self.token_to_ix, self.max_token)

        return img_feat_iter, que, label

