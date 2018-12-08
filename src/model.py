#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models


class img2img(nn.Module):
    def __init__(self, num_ingred=10, imfeatDim=2048, embDim=1024):
        super(img2img, self).__init__()
        self.num_ingred = num_ingred
        self.imfeatDim = imfeatDim
        self.embDim = embDim

        # get resnet50 pretrained model
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        resnet_modules = list(resnet.children())[:-1]
        self.visionMLP = nn.Sequential(*resnet_modules)
        self.visual_embedding = nn.Sequential(nn.Linear(imfeatDim, embDim), nn.Tanh())
        self.ingred_embedding = nn.Sequential(nn.Linear(embDim * num_ingred, embDim), nn.Tanh())

    def forward(self, dish_img, ingred_imgs):
        # dish_img:     size (bsz, 3, 224, 224)
        # ingred_imgs:  size (bsz, 10, 3, 224, 224)
        assert len(ingred_imgs.size()) == 5

        # dish embedding
        dish_feat = self.visionMLP(dish_img)                # size (bsz, 2048, 1, 1)
        dish_feat = dish_feat.view(dish_feat.size(0), -1)   # size (bsz, 2048)
        dish_emb = self.visual_embedding(dish_feat)         # size (bsz, 1024)

        # individual ingredient embedding
        bsz, num_ingred, c, h, w = ingred_imgs.size()
        ingred_feats = self.visionMLP(ingred_imgs.view(-1, c, h, w))    # size (bsz * 10, 2048, 1, 1)
        ingred_feats = ingred_feats.view(ingred_feats.size(0), -1)      # size (bsz * 10, 2048)
        ingred_embs = self.visual_embedding(ingred_feats)               # size (bsz * 10, 1024)

        # concatenate the ingredient embeddings
        concat_ingred = ingred_embs.view(bsz, -1)                       # size (bsz, 1024 * 10)
        ingred_embs = self.ingred_embedding(concat_ingred)              # size (bsz, 1024)

        output = [dish_emb, ingred_embs]
        return output


if __name__ == "__main__":
    net = img2img()
    bsz = 5
    dish_emb, ingred_embs = net(torch.randn(bsz, 3, 224, 224), torch.randn(bsz, 10, 3, 224, 224))
