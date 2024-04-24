import numpy as np
import os
import torch
import torch.utils.data as data
import en_vectors_web_lg
import json, glob



def proc_ques(labels, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    words = labels
    
    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix

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


class Batch_generator_prototype(data.Dataset):
	def __init__(self, mode='train', token_to_idx=None):
		self.label = np.load('/home/lxq/data/fsl_vqa/VQA_v2/label.npy', allow_pickle=True).item()
		self.obj2idx = np.load('/home/lxq/data/fsl_vqa/VQA_v2/obj2idx.npy', allow_pickle=True).item()
		path = glob.glob('/home/lxq/data/fsl_vqa/VQA_v2/vinvl/' + '*.npz')
		self.img_path = []
		
		if mode == 'train':
			datas = torch.load('/home/lxq/data/fsl_vqa/VQA_v2/train.pth')
			self.token_to_ix, self.pretrained_emb = tokenize(list(datas['all_words'].keys()) + list(self.obj2idx.keys()))
			for i in range(int(len(path) * 0.8)):
				self.img_path.append(path[i])
		else:
			self.token_to_ix = token_to_idx
			for i in range(int(len(path) * 0.8), len(path)):
				self.img_path.append(path[i])
		self.token_size = len(self.token_to_ix)

	def __getitem__(self, index):
		path = self.img_path[index]
		id = int(path.split('/')[-1].split('_')[-1].split('.')[0])
		
		label = self.label[str(id)]

		# load image features
		img = np.load(path)
		img = img['x'].transpose((1, 0))
		img = proc_img_feat(img, 75)

		# standard multi-label classification setting
		mask = torch.zeros(len(self.obj2idx))
		for obj in label:
			mask[self.obj2idx[obj]] = 1

		que = proc_ques(label, self.token_to_ix, 75)

		return img, mask, que

	def __len__(self,):
		return len(self.img_path)
