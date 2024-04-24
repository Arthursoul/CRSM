import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import Batch_generator_prototype
from module import Prototype_Module
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import numpy as np
import cv2
import argparse
import os
import time
import gc
import tensorflow as tf
from loss import bce_loss
import json
import tqdm


parser = argparse.ArgumentParser(description='Prototype learning experiments')
parser.add_argument('--mode', type=str, default='vqa', help='Selecting running mode (default: vqa, gqa, or novelvqa)')
parser.add_argument('--checkpoint_dir',type=str, default='./prototype_ckpts', help='Directory for saving checkpoint')
parser.add_argument('--epoch',type=int, default=60, help='Defining maximal number of epochs')
parser.add_argument('--lr',type=float, default=4e-4, help='Defining initial learning rate (default: 4e-4)')
parser.add_argument('--batch_size',type=int, default=64, help='Defining batch size for training (default: 150)')
parser.add_argument('--clip',type=float, default=0.1, help='Gradient clipping to prevent gradient explode (default: 0.1)')
parser.add_argument('--num_proto',type=int, default=3000, help='Number of prototype')

args = parser.parse_args()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(init_lr,optimizer, epoch):
    "adatively adjust lr based on epoch"
    lr = init_lr * (0.25 ** int((epoch+1)/20)) #previously 0.25/10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_dir)

    
    train_data = Batch_generator_prototype('train')
    val_data = Batch_generator_prototype('val', train_data.token_to_ix)
    

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=12)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = Prototype_Module(2048,args.num_proto,len(train_data.obj2idx), train_data.token_size, train_data.pretrained_emb)
    model = model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) #1e-8

    def train(iteration):
        model.train()
        avg_loss = 0
        for batch_idx,(img, label,label_ix) in enumerate(trainloader):
            img, label, label_ix = Variable(img), Variable(label), Variable(label_ix)
            img, label, label_ix = img.cuda(), label.cuda(), label_ix.cuda()
            optimizer.zero_grad()

            prediction = model(img, label_ix)
            loss = bce_loss(prediction,label)
            loss.backward()

            if not args.clip == 0 :
                clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()
            avg_loss = (avg_loss*np.maximum(0,batch_idx) + loss.data.cpu().numpy())/(batch_idx+1)

            if batch_idx%25 == 0:
                with tf_summary_writer.as_default():
                    tf.summary.scalar('average classification loss',avg_loss,step=iteration)
                print('训练阶段:epo {} avg_loss={:.4f}'.format(iteration, avg_loss.item()))

        iteration += 1

        return iteration

    def test(iteration):
        model.eval()
        tp = 0
        fp = 0
        pos = 0
        for batch_idx,(img, label,label_ix) in enumerate(valloader):
            img, label, label_ix = Variable(img), Variable(label), Variable(label_ix)
            img, label, label_ix = img.cuda(), label.cuda(), label_ix.cuda()
            prediction = model(img, label_ix)

            prediction  = prediction.data.cpu().numpy()
            prediction[prediction>=0.5] = 1
            prediction[prediction<0.5] = 0
            label = label.data.cpu().numpy()

            pos += np.sum(label)
            tp += np.sum(label*prediction)
            fp += np.sum((1-label)*prediction)
            precision = tp/(tp+fp)
            recall = tp/pos
            print('验证阶段:epo {} Precision={:.4f} Recall={:.4f}  F1-score={:.4f}'.format(iteration, (tp/(tp+fp)).item(), (tp/pos).item(), (2*(precision*recall)/(precision+recall)).item()))

        precision = tp/(tp+fp)
        recall = tp/pos
        f1_score = 2*(precision*recall)/(precision+recall)


        with tf_summary_writer.as_default():
            tf.summary.scalar('Precision',precision,step=iteration)
            tf.summary.scalar('Recall',recall,step=iteration)
            tf.summary.scalar('F1-score',f1_score,step=iteration)
        
        return f1_score


    #main loop for training:
    print('Start training model')
    iteration = 0
    val_score = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr,optimizer, epoch)
        iteration = train(iteration)
        cur_score = test(iteration)
        #save the best check point and latest checkpoint
        if cur_score > val_score:
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model_best.pth'))
            val_score = cur_score
        torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model.pth'))

main()
