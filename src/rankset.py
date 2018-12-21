from __future__ import print_function

import os
import random
import lmdb
import torch
import pickle
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms


def default_loader(path):
    try:
        return Image.open(path).convert('RGB')
    except:
        print("Failed to Load Image")
        return Image.new('RGB', (224, 224), 'white')


class FeatDataset(data.Dataset):
    def __init__(self,dish_feat_path,ingr_feat_path, dish_info_path,  vocab_path, partition, num_images=10):

        self.env = lmdb.open(os.path.join(dish_info_path, partition, partition + '_lmdb'),
                             max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with open(os.path.join(dish_info_path, partition, partition + '_keys.pkl'), 'rb') as f:
            self.dish_ids = pickle.load(f)


        self.ids = [0]

        self.dish_feat = pickle.load(open(dish_feat_path,'rb'))
        self.ingr_feat = pickle.load(open(ingr_feat_path,'rb'))
        self.vocab = [line.strip() for line in open(vocab_path)]
        self.partition = partition
        self.num_images = num_images


    def set_ids(self,n_neg = 10):
        ids = []
        all_ids = np.arange(len(self.dish_ids))
        random.shuffle(all_ids)
        for i in range(len(self.dish_ids)):
            #ids.append([i,i,1])
            neg_id = np.random.choice(all_ids,n_neg)
            for j in neg_id:
                if i != j:
                    if np.random.rand() > 0.0:
                       ids.append([i,i,j,1.0])
                    else:
                       ids.append([i,j,i,-1.0]) 
            #break
        random.shuffle(ids)
        self.ids = ids
                

    def __getitem__(self, index):

        # Load dish image
        inst = self.ids[index]
        dish1_img_feat = torch.FloatTensor(self.dish_feat[inst[1]]['feat'])
        dish2_img_feat = torch.FloatTensor(self.dish_feat[inst[2]]['feat'])
        ingr_list = self.dish_feat[inst[0]]['ingrs']
        target = inst[3]
        #print(dish_img_feat)
        ingr_img_feat = torch.zeros(10,len(dish1_img_feat))
        #print(ingr_img_feat.shape)
        nn = 0
        for i in ingr_list:
            ingr_id = self.vocab[i-2]
            if ingr_id in self.ingr_feat:
                ingr_img_feat[nn] = torch.FloatTensor(random.sample(self.ingr_feat[ingr_id],1)[0])
                nn += 1
        #print(ingr_img_feat.shape,dish_img_feat.shape)
        return [ingr_img_feat,dish1_img_feat,dish2_img_feat],target



    def __len__(self):
        return len(self.ids)



def tensor_to_numpy(x):
    """ convert tensor within [0, 1], shape (3, 224, 224) to numpy within [0, 255], shape (224, 224, 3). """
    return (x.permute(1, 2, 0).data.numpy() * 225).astype(int)


if __name__ == "__main__":
    val_dataset = ImageDataset(dish_img_path="../data/dish_img", dish_info_path='../data/dish_info',
                               ingr_img_path='../data/ingr_img', vocab_path='../data/vocab.txt',
                               partition='val',
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])]))
    [dish_img, ingr_imgs, igr_ln], [target, img_id, rec_id] = val_dataset[4]

    import matplotlib.pyplot as plt
    # show dish image
    plt.imshow(tensor_to_numpy(dish_img))
    plt.show()

    # show ingredient images
    for i in range(10):
        plt.clf()
        plt.imshow(tensor_to_numpy(ingr_imgs[i]))
        plt.show()
