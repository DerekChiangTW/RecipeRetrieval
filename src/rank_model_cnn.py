#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models


class img2img_feat(nn.Module):
    def __init__(self, num_ingr=3, imfeatDim=2048, embDim=1024):
        super(img2img_feat, self).__init__()
        self.num_ingr = num_ingr
        self.imfeatDim = imfeatDim
        self.embDim = embDim

        # get resnet50 pretrained model
        #self.visual_embedding1 = nn.Sequential(nn.Linear(imfeatDim, embDim), nn.Tanh())
        self.visual_embedding1 = nn.Sequential(nn.Linear(imfeatDim, embDim), nn.Tanh(),nn.Dropout(0.2),nn.Linear(embDim, embDim), nn.Tanh())
        #self.ingr_embedding = nn.Sequential(nn.Linear(embDim * num_ingr, embDim), nn.Tanh())
        #self.ingr_embedding = nn.Sequential(nn.Linear(imfeatDim * num_ingr, embDim), nn.Tanh())
        #self.ingr_embedding = nn.Sequential(nn.Linear(imfeatDim * num_ingr, embDim), nn.Tanh(),nn.Dropout(0.2),nn.Linear(embDim,embDim),nn.Tanh())
        self.ingr_embedding = nn.Sequential(nn.Conv2d(1,embDim,(3,imfeatDim)), nn.Tanh(),nn.MaxPool2d(kernel_size=(8,1)))
        self.ingr_embedding2 = nn.Sequential(nn.Linear(embDim,embDim),nn.Tanh())
        self.score = nn.Sequential(nn.Linear(2*embDim,embDim),nn.Tanh(),nn.Dropout(0.2),nn.Linear(embDim,1))
        #self.score2 = nn.Sequential(nn.Linear(2*imfeatDim,embDim),nn.ReLU(),nn.Linear(embDim,1))

    def forward(self, dish_feat, ingr_feats):
        # dish_img:   size (batch_size, 3, 224, 224)
        # ingr_imgs:  size (batch_size, num_ingr, 3, 224, 224)

        # dish embedding
        dish_emb = self.visual_embedding1(dish_feat)         # size (bsz, 1024)

        # individual ingredient embedding
        #'''
        #ingr_embs = self.visual_embedding2(ingr_feats)               # size (bsz * num_ingr, 1024)
        ingr_embs = ingr_feats.clone()
        dims = ingr_embs.shape
        ingr_embs = ingr_embs.view(dims[0],1,dims[1],dims[2])
        # concatenate the ingredient embeddings
        #concat_ingr = ingr_embs.view(ingr_embs.size(0), -1)           # size (bsz, 1024 * 10)
        ingr_embs = self.ingr_embedding(ingr_embs).squeeze() # size (bsz, 1024)
        ingr_embs = self.ingr_embedding2(ingr_embs)
        #'''
        #ingr_embs = self.visual_embedding2(ingr_feats)
        final_emb = torch.cat((dish_emb,ingr_embs),1)
        #final_emb = torch.cat((dish_feat,ingr_feats),1)
        output = self.score(final_emb)
        return output




if __name__ == "__main__":
    net = img2img()
    bsz = 5
    dish_emb, ingr_embs = net(torch.randn(bsz, 3, 224, 224), torch.randn(bsz, 10, 3, 224, 224))
