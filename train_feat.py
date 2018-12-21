#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
#from src.feat_model import *
#from src.featset import *
from src.rank_model_cos import *
from src.rankset import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_ingr", type=int, default=10)
    parser.add_argument("--imfeatDim", type=int, default=2048)
    parser.add_argument("--embDim", type=int, default=1024)
    parser.add_argument("--dish_feat_path", type=str, default="data/")
    parser.add_argument("--ingr_feat_path", type=str, default="data/")
    parser.add_argument("--dish_info_path", type=str, default="data/dish_info")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.txt")
    args = parser.parse_args()
    return args


def train(opt):
    part = 'val'
    training_set = FeatDataset(opt.dish_feat_path+part+'_featset.pkl', opt.ingr_feat_path+'ingr_feat.pkl',opt.dish_info_path,opt.vocab_path,part)
    #training_set = ImageDataset(opt.dish_img_path, opt.dish_info_path, opt.ingr_img_path, opt.vocab_path,
    #                            'val', transform=transform)
    #testing_set = ImageDataset(opt.dish_img_path, opt.dish_info_path, opt.ingr_img_path, opt.vocab_path,
    #                           'test', transform=transform)
    #train_loader = data.DataLoader(training_set, batch_size=opt.batch_size, shuffle=True)
    #val_loader = data.DataLoader(validate_set, batch_size=opt.batch_size, shuffle=False)
    #test_loader = data.DataLoader(testing_set, batch_size=opt.batch_size, shuffle=False)

    # set the model
    model = img2img_feat(num_ingr=opt.num_ingr, imfeatDim=opt.imfeatDim, embDim=opt.embDim)
    model = model.to(device)
    # set the optimizer
    #criterion = nn.CosineEmbeddingLoss(0.1).to(device)
    criterion = nn.MarginRankingLoss(margin=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    #num_iter_per_epoch = len(train_loader)

    training_set.set_ids()
    model.train()
    for epoch in range(opt.num_epochs):
        #training_set.set_ids()
        train_loader = data.DataLoader(training_set, batch_size=opt.batch_size, shuffle=False)
        num_iter_per_epoch = len(train_loader)
        print(epoch)
        #print(training_set.ids)
        for itr, (input, target) in enumerate(train_loader):
            #[ingr_imgs, dish_img] = [record.to(device) for record in input]
            [ingr_imgs, dish_img1,dish_img2] = [record.to(device) for record in input]
            target_t = target.type(torch.FloatTensor)
            target_t = target_t.to(device)
            optimizer.zero_grad()
            #[dish_emb, ingr_embs] = model(dish_img, ingr_imgs)
            s1 = model(dish_img1, ingr_imgs)
            s2 = model(dish_img2, ingr_imgs)
            #s1 = model(dish_img1, dish_img1)
            #s2 = model(dish_img2, dish_img1)
            #print(s1)
            #print(s1.shape)
            #loss = criterion(dish_emb, ingr_embs, target_t)
            loss = criterion(s1, s2, target_t)
            loss.backward()
            optimizer.step()
            print("Training: Iteration: {}/{} Epoch: {}/{} Loss: {}".format(
                itr + 1, num_iter_per_epoch, epoch + 1, opt.num_epochs, loss))
            sys.stdout.flush()
        if (epoch+1) % 5 == 0:
            torch.save(model,'model/rank_2emb_half_cos_%d.pth' % (epoch))

def eval(model, opt):
    pass


def main():
    opt = get_args()
    train(opt)


if __name__ == "__main__":
    main()
