#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pickle
import argparse
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
from src.model import *
from src.imagedata import *
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_ingr", type=int, default=10)
    parser.add_argument("--imfeatDim", type=int, default=2048)
    parser.add_argument("--embDim", type=int, default=1024)
    parser.add_argument("--dish_img_path", type=str, default="data/dish_img")
    parser.add_argument("--dish_info_path", type=str, default="data/dish_info")
    parser.add_argument("--ingr_img_path", type=str, default="data/ingr_img")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.txt")
    args = parser.parse_args()
    return args


def train(opt):

    # set image transformation methods
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),     # we get only the center of that rescaled
        transforms.RandomCrop(224),     # random crop within the center crop
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # prepare the train, validate and test loader
    #training_set = ImageDataset(opt.dish_img_path, opt.dish_info_path, opt.ingr_img_path, opt.vocab_path,
     #                           'train', transform=transform)
    data_set = ImageData(opt.dish_img_path, opt.dish_info_path, opt.ingr_img_path, opt.vocab_path,
                                'test', transform=transform)
    #testing_set = ImageDataset(opt.dish_img_path, opt.dish_info_path, opt.ingr_img_path, opt.vocab_path,
    #                           'test', transform=transform)
    #train_loader = data.DataLoader(training_set, batch_size=opt.batch_size, shuffle=True)
    #val_loader = data.DataLoader(validate_set, batch_size=opt.batch_size, shuffle=False)
    #test_loader = data.DataLoader(testing_set, batch_size=opt.batch_size, shuffle=False)
    data_loader = data.DataLoader(data_set, batch_size=opt.batch_size, shuffle=True)

    # set the model
    #model = img2img(num_ingr=opt.num_ingr, imfeatDim=opt.imfeatDim, embDim=opt.embDim)
    #model = model.to(device)
    model = models.resnet50(pretrained=True)
    resnet_modules = list(model.children())[:-1]
    model = nn.Sequential(*resnet_modules)
    model = model.to(device)

    # set the optimizer
    criterion = nn.CosineEmbeddingLoss(0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    num_iter_per_epoch = len(data_loader)
    print( num_iter_per_epoch)

    model.eval()
    rec = {}
    for epoch in range(opt.num_epochs):
        for itr, input in enumerate(data_loader):
            #[dish_img, ingr_imgs, igr_ln] = [record.to(device) for record in input]
            #[dish_img, ingr_imgs, igr_ln] = [record.to(device) for record in input]
            dish_img = input[0].to(device)
            idx = input[1].numpy()
            #target = [record.to(device) for record in target]
            #optimizer.zero_grad()
            dish_emb  = model(dish_img)
            dish_emb = dish_emb.view(dish_emb.size(0),-1).cpu().detach().numpy()
            for i in range(len(idx)):
                rec[idx[i]] = dish_emb[i]
            print(len(rec))
            sys.stdout.flush()
            #loss = criterion(dish_emb, ingr_embs, target[0])
            #loss.backward()
            #optimizer.step()
            #print("Training: Iteration: {}/{} Epoch: {}/{} Loss: {}".format(
            #    itr + 1, num_iter_per_epoch, epoch + 1, opt.num_epochs, loss))
    pickle.dump(rec,open('test_feat2.pkl','wb'))

def eval(model, opt):
    pass


def main():
    opt = get_args()
    train(opt)


if __name__ == "__main__":
    main()
