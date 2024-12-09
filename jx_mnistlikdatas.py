import numpy as np
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch.utils.data as data
import torch
import os.path
import matplotlib.image as mpimg 
import numpy as np
import torch
# from fastai.basics import *
import re

def get_file_path(root_path, file_list, dir_list):
    dir_or_files = os.listdir(root_path)
    dir_or_files.sort()
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)

def jx_getfilelist(root_dir):
    size = 0
    all_file_list = []
    dir_list = []
    get_file_path(root_dir, all_file_list, dir_list)
    return all_file_list, dir_list

def savedataset(all_file_list, Savepath):
    ii = 0
    number = len(all_file_list)
    allimArr = torch.Tensor(number, 64, 64)
    alllabel = torch.Tensor(number)
    for sglfilename in all_file_list:
        pattern = r'[/]'  
        splited = re.split(pattern, sglfilename) 
        imname = splited[-1]
        if imname[-4:] == '.jpg' or imname[-4:] == '.bmp':
            label = splited[-2]
            imArr = mpimg.imread(sglfilename)
            imArr = torch.from_numpy(imArr) 
            allimArr[ii, :, :] = imArr
            lbArr = int(label)
            lbArr = torch.from_numpy(np.array(lbArr))
            alllabel[ii] = lbArr
            ii += 1
    ##########################
    Savedataset = (allimArr, alllabel)
    torch.save(Savedataset, Savepath)
    return Savedataset

class jx_phototourlike(data.Dataset):
    def __init__(self, root, name, train=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.data_file = os.path.join(self.root, '{}.pt'.format(name))
        self.transform = None
        self.target_transform = None

        if os.path.exists(self.data_file):
            print('# Found cached data {}'.format(self.data_file))
            self.data, self.labels = torch.load( os.path.join(self.data_file) )
        else:
            tmpdir = os.path.join(self.root,name)
            all_file_list, dir_list = jx_getfilelist( tmpdir )
            self.data, self.labels = savedataset(all_file_list, self.data_file)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class SiameseMNIST_subset(Dataset):
    
    def __init__(self, sub_dataset,sub_indices):
        self.indices = sub_indices
        self.train = sub_dataset.train
        self.transform = sub_dataset.transform
        self.data_len = 0
        if self.train:
            self.train_labels = sub_dataset.train_labels[self.indices]
            self.train_data = sub_dataset.train_data[self.indices]
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0] for label in self.labels_set}
            self.data_len = len(self.train_data)
        else:
            self.test_labels = sub_dataset.test_labels[self.indices]
            self.test_data = sub_dataset.test_data[self.indices]
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0] for label in self.labels_set}
            random_state = np.random.RandomState(29)
            positive_pairs = [ [i, random_state.choice(self.label_to_indices[self.test_labels[i].item()]),  1]
                               for i in range(0, len(self.test_data), 2) ]
            negative_pairs = [ [i, random_state.choice(self.label_to_indices[ np.random.choice( list(self.labels_set - set( [self.test_labels[i].item()]) ) ) ] ), 0]
                               for i in range(1, len(self.test_data), 2) ]
            self.test_pairs = positive_pairs + negative_pairs
            self.data_len = len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label] )
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return self.data_len

class TripletMNIST_subet(Dataset):
       def __init__(self, sub_dataset,sub_indices):
        self.indices = sub_indices
        self.train = sub_dataset.train
        self.transform = sub_dataset.transform
        self.data_len = 0
        if self.train:
            self.train_labels = sub_dataset.train_labels[self.indices]
            self.train_data = sub_dataset.train_data[self.indices]
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0] for label in self.labels_set}
            self.data_len = len(self.train_data)
        else:
            self.test_labels = sub_dataset.test_labels[self.indices]
            self.test_data = sub_dataset.test_data[self.indices]
            
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]  for label in self.labels_set}
            random_state = np.random.RandomState(29)
            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[np.random.choice( list(self.labels_set - set([self.test_labels[i].item()])) ) ])
                        ] for i in range(len(self.test_data))]
            self.test_triplets = triplets
            self.data_len = len(self.test_data)
    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return self.data_len ##len(self.mnist_dataset)

class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=True)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
