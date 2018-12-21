#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import random
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
#from src.feat_model import *
#from src.featset import *
from src.rank_model import *
from src.rankset import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_eval(model,testing_set,cnn=False):
    dish_feat = []
    ingr_feat = []
    #outfile = open('rank.rec','w')
    #for i in range(len(testing_set.dish_feat)):
    ind = np.arange(len(testing_set.dish_feat))
    random.shuffle(ind)
    trec = {}
    rrec = ind[:1000]
    for i in ind[:1000]:
        trec[i] = {'ingr':[],'ranklist':[]}
        dish_img_feat = testing_set.dish_feat[i]['feat']
        dish_feat.append(dish_img_feat)
        ingr_img_feat = np.zeros((10,len(dish_img_feat)))
        nn = 0
        for j in testing_set.dish_feat[i]['ingrs']:
             ingr_id = testing_set.vocab[j-2]
             if ingr_id in testing_set.ingr_feat:
                trec[i]['ingr'].append(ingr_id)
                ingr_img_feat[nn] = random.sample(testing_set.ingr_feat[ingr_id],1)[0]
                nn += 1
        ingr_feat.append(ingr_img_feat)
    dish_feat = np.array(dish_feat)
    dish_emb = torch.FloatTensor(dish_feat).to(device)
    dish_emb = model.visual_embedding1(dish_emb)

    ingr_feat = np.array(ingr_feat)
    ingr_embs = torch.FloatTensor(ingr_feat).to(device)

    if cnn:
        dims = ingr_embs.shape
        ingr_embs = ingr_embs.view(dims[0],1,dims[1],dims[2])
        ingr_embs = model.ingr_embedding(ingr_embs).squeeze() # size (bsz, 1024)
        ingr_embs = model.ingr_embedding2(ingr_embs)
    else:
        concat_ingr = ingr_embs.view(ingr_embs.size(0), -1)           # size (bsz, 1024 * 10)
        ingr_embs = model.ingr_embedding(concat_ingr)

    rec = []
    for i in range(1000):
        ingr_in = ingr_embs[i].repeat(1000,1)
        final_emb = torch.cat((dish_emb,ingr_in),1)
        score = model.score(final_emb)
        #score = model.score(dish_emb,ingr_in)
        score = score.cpu().detach().numpy().flatten()
        rank_list = score.argsort()[::-1]
        for r in rank_list:
            trec[rrec[i]]['ranklist'].append(rrec[r])
        rank = np.where(rank_list==i)
        trec[rrec[i]]['rank'] = rank
        #print(rank)
        rec.append(rank)
    rec = np.array(rec)
    #print(rec.mean())
    #print(np.median(rec))
    #print((rec < 10).sum()/1000)
    pickle.dump(trec,open('/home/ylien/case.rec','wb'))
    return np.median(rec),(rec < 1).sum()/1000,(rec < 5).sum()/1000,(rec < 10).sum()/1000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dish_feat_path", type=str, default="data/")
    parser.add_argument("--ingr_feat_path", type=str, default="data/")
    parser.add_argument("--dish_info_path", type=str, default="data/dish_info")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.txt")
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()
    return args


def test(opt):
    #part = 'test'
    part = 'test'
    testing_set = FeatDataset(opt.dish_feat_path+part+'_featset.pkl', opt.ingr_feat_path+'ingr_feat.pkl',opt.dish_info_path,opt.vocab_path,part)
    model = torch.load(opt.model_path)
    model.eval()
    metric = [[],[],[],[]]
    for _ in range(1):
        medR,r1,r5,r10 = model_eval(model,testing_set,('cnn' in opt.model_path))
        metric[0].append(medR)
        metric[1].append(r1)
        metric[2].append(r5)
        metric[3].append(r10)
    metric = np.array(metric)
    for i in range(4):
        print(metric[i].mean())



def main():
    opt = get_args()
    test(opt)


if __name__ == "__main__":
    main()
