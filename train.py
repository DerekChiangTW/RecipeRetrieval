#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from src.model import *
from src.dataset import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=10)
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
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # prepare the train, validate and test loader
    training_set = ImageDataset(opt.dish_img_path, opt.dish_info_path, opt.ingr_img_path, opt.vocab_path,
                                'train', transform=transform)
    validate_set = ImageDataset(opt.dish_img_path, opt.dish_info_path, opt.ingr_img_path, opt.vocab_path,
                                'val', transform=transform)
    testing_set = ImageDataset(opt.dish_img_path, opt.dish_info_path, opt.ingr_img_path, opt.vocab_path,
                               'test', transform=transform)
    train_loader = data.DataLoader(training_set, batch_size=opt.batch_size, shuffle=True)
    val_loader = data.DataLoader(validate_set, batch_size=opt.batch_size, shuffle=False)
    test_loader = data.DataLoader(testing_set, batch_size=opt.batch_size, shuffle=False)

    # set the model
    model = img2img(num_ingr=opt.num_ingr, imfeatDim=opt.imfeatDim, embDim=opt.embDim)
    model = model.to(device)

    # set the optimizer
    criterion = nn.CosineEmbeddingLoss(0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    num_iter_per_epoch = len(train_loader)

    model.train()
    for epoch in range(opt.num_epochs):
        for itr, (input, target) in enumerate(train_loader):
            [dish_img, ingr_imgs, igr_ln] = [record.to(device) for record in input]
            target = [record.to(device) for record in target]

            optimizer.zero_grad()
            [dish_emb, ingr_embs] = model(dish_img, ingr_imgs)
            loss = criterion(dish_emb, ingr_embs, target[0])
            loss.backward()
            optimizer.step()
            print("Training: Iteration: {}/{} Epoch: {}/{} Loss: {}".format(
                itr + 1, num_iter_per_epoch, epoch + 1, opt.num_epochs, loss))


def eval(model, opt):
    pass


def main():
    opt = get_args()
    train(opt)


if __name__ == "__main__":
    main()
