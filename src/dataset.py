from __future__ import print_function
from PIL import Image

import os
import lmdb
import torch
import pickle
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import json


def default_loader(path):
    try:
        return Image.open(path).convert('RGB')
    except:
        print("Failed to Load Image : ", path)
        return Image.new('RGB', (224, 224), 'white')


class ImageDataset(data.Dataset):
    def __init__(self, dish_img_path, dish_info_path, ingr_img_path, vocab_path, title_path, partition, transform=None,
                 target_transform=None, loader=default_loader, square=False, num_images=10):

        self.env = lmdb.open(os.path.join(dish_info_path, partition, partition + '_lmdb'),
                             max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with open(os.path.join(dish_info_path, partition, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)

        self.dish_img_path = dish_img_path
        self.ingr_img_path = ingr_img_path
        self.vocab_path = vocab_path
        self.partition = partition
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.square = square
        self.num_images = num_images
        self.mismtch = 0.8
        self.maxInst = 20
        with open(title_path) as json_file:
            self.title_map = json.load(json_file)

    def __getitem__(self, index):
        # we force 80 percent of them to be a mismatch
        if self.partition == 'train':
            match = np.random.uniform() > self.mismtch
        elif self.partition == 'val' or self.partition == 'test':
            match = True
        else:
            raise 'Partition name not well defined'

        target = match and 1 or -1

        # Load dish image
        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index])
        sample = pickle.loads(serialized_sample)
        imgs = sample['imgs']

        # image
        if target == 1:
            if self.partition == 'train':
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(imgs))))
            else:
                imgIdx = 0

            loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
            loader_path = os.path.join(*loader_path)
            path = os.path.join(self.dish_img_path, self.partition, loader_path, imgs[imgIdx]['id'])
        else:
            # we randomly pick one non-matching image
            all_idx = range(len(self.ids))
            rndindex = np.random.choice(all_idx)
            while rndindex == index:
                rndindex = np.random.choice(all_idx)  # pick a random index

            with self.env.begin(write=False) as txn:
                serialized_sample = txn.get(self.ids[rndindex])

            rndsample = pickle.loads(serialized_sample)
            rndimgs = rndsample['imgs']

            if self.partition == 'train':  # if training we pick a random image
                # We do only use the first five images per recipe during training
                imgIdx = np.random.choice(range(min(5, len(rndimgs))))
            else:
                imgIdx = 0
            # Not sure if the commented line is a bug
            # path = self.dish_img_path + rndimgs[imgIdx]['id']
            loader_path = [rndimgs[imgIdx]['id'][i] for i in range(4)]
            loader_path = os.path.join(*loader_path)
            path = os.path.join(self.dish_img_path, self.partition, loader_path, rndimgs[imgIdx]['id'])

        # instructions
        instrs = sample['intrs']
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        if itr_ln > self.maxInst:
            t_inst[:self.maxInst][:] = instrs[:self.maxInst][:]
        else:
            t_inst[:itr_ln][:] = instrs[:itr_ln][:]
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        ingrs = sample['ingrs'].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        # get ingredient images, tensor size (10, 3, 224, 224)
        ingr_imgs = self.get_ingr_imgs(sample['ingrs'].tolist())

        # load dish image, tensor size (3, 224, 224)
        dish_img = self.loader(path)

        if self.square:
            dish_img = dish_img.resize(self.square)
        if self.transform is not None:
            dish_img = self.transform(dish_img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # rec_class = sample['classes'] - 1
        rec_id = self.ids[index]

        if target == -1:
            # img_class = rndsample['classes'] - 1
            img_id = self.ids[rndindex]
        else:
            # img_class = sample['classes'] - 1
            img_id = self.ids[index]

        # title
        title = self.title_map[rec_id]

        """
        # output
        if self.partition == 'train':
            return [dish_img, ingr_imgs, igr_ln, instrs, itr_ln, title], [target]
        else:
            return [dish_img, ingr_imgs, igr_ln, instrs, itr_ln, title], [target, img_id, rec_id]
        """
        # print("sampluee:")
        # print(sample)
        return [dish_img, ingr_imgs, igr_ln, instrs, itr_ln, title], [target]

    def __len__(self):
        return 4
        # return len(self.ids)

    def get_ingr_imgs(self, indices):
        # read all vocab
        with open(self.vocab_path) as infile:
            vocabs = [line.rstrip() for line in infile]

        # get ingredient names
        split = min(self.num_images, max(np.nonzero(indices)[0]))
        ingr_names = [vocabs[i - 2] for i in indices[:split]]

        # get ingredient images
        ingr_imgs = torch.zeros((self.num_images, 3, 224, 224))
        for i in range(self.num_images):
            if i < len(ingr_names):
                folder = os.path.join(self.ingr_img_path, ingr_names[i])
                path = os.path.join(folder, np.random.choice(os.listdir(folder)))

                # load ingredient image
                img = self.loader(path)
                if self.square:
                    img = img.resize(self.square)
                if self.transform is not None:
                    img = self.transform(img)
                ingr_imgs[i] = img

        return ingr_imgs


def tensor_to_numpy(x):
    """ convert tensor within [0, 1], shape (3, 224, 224) to numpy within [0, 255], shape (224, 224, 3). """
    return (x.permute(1, 2, 0).data.numpy() * 225).astype(int)


if __name__ == "__main__":
    val_dataset = ImageDataset(dish_img_path="../data/dish_img", dish_info_path='../data/dish_info',
                               ingr_img_path='../data/ingr_img', vocab_path='../data/vocab.txt',
                               title_path='../data/title_map.json',
                               partition='val',
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])]))
    [dish_img, ingr_imgs, igr_ln, instrs, itr_ln, title], [target, img_id, rec_id] = val_dataset[4]

    print (title)

    import matplotlib.pyplot as plt

    # show dish image
    plt.imshow(tensor_to_numpy(dish_img))
    plt.show()

    # show ingredient images
    for i in range(10):
        plt.clf()
        plt.imshow(tensor_to_numpy(ingr_imgs[i]))
        plt.show()
