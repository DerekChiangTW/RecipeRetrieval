#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models


class img2img(nn.Module):
    def __init__(self, num_ingr=10, imfeatDim=2048, embDim=1024):
        super(img2img, self).__init__()
        self.num_ingr = num_ingr
        self.imfeatDim = imfeatDim
        self.embDim = embDim

        # get resnet50 pretrained model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        resnet_modules = list(resnet.children())[:-1]
        self.visionMLP = nn.Sequential(*resnet_modules)
        self.visual_embedding = nn.Sequential(nn.Linear(imfeatDim, embDim), nn.Tanh())
        self.ingr_embedding = nn.Sequential(nn.Linear(embDim * num_ingr, embDim), nn.Tanh())

    def forward(self, dish_img, ingr_imgs):
        # dish_img:   size (batch_size, 3, 224, 224)
        # ingr_imgs:  size (batch_size, num_ingr, 3, 224, 224)

        # dish embedding
        dish_feat = self.visionMLP(dish_img)                # size (bsz, 2048, 1, 1)
        dish_feat = dish_feat.view(dish_feat.size(0), -1)   # size (bsz, 2048)
        dish_emb = self.visual_embedding(dish_feat)         # size (bsz, 1024)

        # individual ingredient embedding
        bsz, num_ingr, c, h, w = ingr_imgs.size()
        ingr_feats = self.visionMLP(ingr_imgs.view(-1, c, h, w))    # size (bsz * num_ingr, 2048, 1, 1)
        ingr_feats = ingr_feats.view(ingr_feats.size(0), -1)        # size (bsz * num_ingr, 2048)
        ingr_embs = self.visual_embedding(ingr_feats)               # size (bsz * num_ingr, 1024)

        # concatenate the ingredient embeddings
        concat_ingr = ingr_embs.view(bsz, -1)           # size (bsz, 1024 * 10)
        ingr_embs = self.ingr_embedding(concat_ingr)    # size (bsz, 1024)

        output = [dish_emb, ingr_embs]
        return output


if __name__ == "__main__":
    net = img2img()
    bsz = 5
    dish_emb, ingr_embs = net(torch.randn(bsz, 3, 224, 224), torch.randn(bsz, 10, 3, 224, 224))
