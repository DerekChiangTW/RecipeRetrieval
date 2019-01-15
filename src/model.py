#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models


class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()

    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y

# Skip-thoughts LSTM
class stRNN(nn.Module):
    def __init__(self):
        super(stRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=1024, bidirectional=False, batch_first=True)

    def forward(self, x, sq_lengths):
        # here we use a previous LSTM to get the representation of each instruction
        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx \
            .view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequencea
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the lstm
        out, hidden = self.lstm(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unsorted_idx = original_idx.view(-1, 1, 1).expand_as(unpacked)
        # we get the last index of each sequence in the batch
        idx = (sq_lengths - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        # we sort and get the last element of each sequence
        output = unpacked.gather(0, unsorted_idx.long()).gather(1, idx.long())
        output = output.view(output.size(0), output.size(1) * output.size(2))

        return output



class img2img(nn.Module):
    def __init__(self, num_ingr=10, imfeatDim=2048, embDim=1024, ):
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
        self.inst_embedding = stRNN()
        self.table = TableModule()
        self.recipe_embedding = nn.Sequential(
            #nn.Linear(opts.irnnDim * 2 + opts.srnnDim, opts.embDim, opts.embDim),
            nn.Linear(embDim*2, embDim, embDim),
            nn.Tanh(),
        )

    def forward(self, dish_img, ingr_imgs, instrs, itr_ln):
        # dish_img:   size (batch_size, 3, 224, 224)
        # ingr_imgs:  size (batch_size, num_ingr, 3, 224, 224)
        # instrs:   size (bsz, 20, 1024)
        # itr_ln:   size (bsz)

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

        # instruction embedding
        inst_embs = self.inst_embedding(instrs, itr_ln)  # size (bsz, 1024)

        # recipe embedding
        recipe_emb = self.table([ingr_embs, inst_embs], 1)          # size (bsz, 2048)
        recipe_emb = self.recipe_embedding(recipe_emb)              # size (bsz, 1024)

        output = [dish_emb, recipe_emb]
        return output


if __name__ == "__main__":
    net = img2img()
    bsz = 5
    dish_emb, ingr_embs = net(torch.randn(bsz, 3, 224, 224), torch.randn(bsz, 10, 3, 224, 224))
