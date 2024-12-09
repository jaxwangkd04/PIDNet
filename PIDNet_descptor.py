#!/usr/bin/python2 -utt
#-*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
from copy import deepcopy
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import copy
from EvalMetrics import ErrorRateAt95Recall
from Loggers import Logger, FileLogger
from Utils import L2Norm, cv2_scale
from Utils import str2bool
import torch.utils.data as data
import torch.utils.data as data_utils
import torch.nn.functional as F

import faiss
import jx_networks
from jx_mnistlikdatas import jx_phototourlike
import util_visdom 
import warp
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet')
# Model options
parser.add_argument('--isTesting', type=bool,   default= False,  help='Set True when testing')
parser.add_argument('--test_checkIndx',type=int, default = 0, help= 'Network Params for testing: checkpoint_{}.pth')
parser.add_argument('--dataroot', type=str,   default='../a',   help='path to dataset') 
parser.add_argument('--resume', default='', type=str, metavar='PATH',  help='path to latest checkpoint (default: none)')
parser.add_argument('--enable-logging',type=bool, default=True,  help='folder to output model checkpoints')
parser.add_argument('--log-dir', default='./logs',   help='folder to output model checkpoints')
parser.add_argument('--experiment-name', default= '/b',  help='experiment path')
parser.add_argument('--trainingset', default= '1',  help='Other options:')
parser.add_argument('--num-workers', default= 8,   help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,   help='')
parser.add_argument('--anchorave', type=bool, default=False,   help='anchorave')
parser.add_argument('--imageSize', type=int, default=64,   help='the height / width of the input image to network')
parser.add_argument('--mean-image', type=float, default=0.443728476019,   help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,  help='std of train dataset for normalization')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',  help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=100, metavar='E',  help='number of epochs to train (default: 10)')
parser.add_argument('--anchorswap', type=bool, default=True, help='turns on anchor swap')
parser.add_argument('--batch_size', type=int, default=256, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='BST',  help='input batch size for testing (default: 1000)')
parser.add_argument('--n-triplets', type=int, default=50000, metavar='N', 
                    help='how many triplets will generate from the dataset')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',   help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--act-decay', type=float, default=0,    help='activity L2 decay, default 0')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',   help='learning rate (default: 0.1)')
parser.add_argument('--fliprot', type=str2bool, default=False,   help='turns on flip and 90deg rotation augmentation')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',   help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,   metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,        help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',   help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')
parser.add_argument("--warpType",	default="homography",	help="type of warp function on images")
parser.add_argument("--size",		default="64x64",		help="image resolution")
parser.add_argument("--total_corners_mat", help="points coordinate file for Homography calculation",
                    default="./total_mat.npy")
parser.add_argument("--port",		type=int,	default=8097,	help="port number for visdom visualization")
parser.add_argument("--pertScale",	type=float,	default=0.25,	help="initial perturbation scale")
parser.add_argument("--transScale",	type=float,	default=0.25,	help="initial translation scale")

args = parser.parse_args()
args.H,args.W = [int(x) for x in args.size.split("x")]
args.canon4pts = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float32)
args.image4pts = np.array([[0, 0], [0, args.H - 1], [args.W - 1, args.H - 1], [args.W - 1, 0]], dtype=np.float32)
args.refMtrx = np.eye(3).astype(np.float32)
dataset_names = ['1', '2', '3']
# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

LOG_DIR = args.log_dir + args.experiment_name
# create loggin directory
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
# set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

class TripletCities(jx_phototourlike):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, root, name, train=True, transform=None, batch_size = None, *arg, **kw):
        super(TripletCities, self).__init__(root, name)
        self.transform = transform
        self.train = train
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size
        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)
        else:
            self.labels_set = set(self.labels.numpy())
            self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0] for label in self.labels_set}
            random_state = np.random.RandomState(29)
            positive_pairs = [[i, random_state.choice(self.label_to_indices[self.labels[i].item()]), 1]
                              for i in range(0, len(self.data), 2)]
            negative_pairs = [[i, random_state.choice(
                self.label_to_indices[np.random.choice(list(self.labels_set - set([self.labels[i].item()])))]), 0]
                              for i in range(1, len(self.data), 2)]
            self.test_pairs = positive_pairs + negative_pairs
    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind.item() not in inds:
                    inds[ind.item()] = []
                inds[ind.item()].append(idx)
            return inds
        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes )
            c1_lab = unique_labels[c1]
            while c1_lab in already_idxs :
                if len(already_idxs) >= n_classes:
                    break
                c1 = np.random.randint(0, n_classes )
                c1_lab = unique_labels[c1]
            already_idxs.add(c1_lab)
            c2 = np.random.randint(0, n_classes )
            c2_lab = unique_labels[c2]
            while c1_lab == c2_lab:
                c2 = np.random.randint(0, n_classes )
                c2_lab = unique_labels[c2]
            keyarrs = indices.get(c1_lab,-1)
            if len(keyarrs) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1_lab]) - 1)
                n2 = np.random.randint(0, len(indices[c1_lab]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1_lab]) - 1)
            n3 = np.random.randint(0, len(indices[c2_lab]) - 1)
            triplets.append([indices[c1_lab][n1], indices[c1_lab][n2], indices[c2_lab][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            img1 = self.data[self.test_pairs[index][0]]
            img2 = self.data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
            img1,img2 = transform_img(img1),transform_img(img2)
            return img1, img2, target

        t = self.triplets[index]
        a, p, n = self.train_data[t[0]], self.train_data[t[1]], self.train_data[t[2]]
        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = transform_img(n)
        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return len(self.test_pairs)

class TripletPhotoTourHardNegatives(jx_phototourlike):##
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self,root, name, negative_indices, train=True, transform=None, batch_size = None, *arg, **kw):
        super(TripletPhotoTourHardNegatives, self).__init__(root, name, *arg, **kw)
        self.transform = transform
        self.train = train
        self.n_triplets = args.n_triplets
        self.negative_indices = negative_indices
        self.batch_size = batch_size
        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets, self.negative_indices)

    @staticmethod
    def generate_triplets(labels, num_triplets, negative_indices):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind.item() not in inds:
                    inds[ind.item()] = []
                inds[ind.item()].append(idx)
            return inds
        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        already_idxs = set()
        count  = 0
        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes )
            c1_lab = unique_labels[c1]
            while c1_lab in already_idxs:
                if len(already_idxs) >= n_classes:
                    break
                c1 = np.random.randint(0, n_classes)
                c1_lab = unique_labels[c1]
            already_idxs.add(c1_lab)
            if len(indices[c1_lab]) == 2:
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1_lab]))
                n2 = np.random.randint(0, len(indices[c1_lab]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1_lab]))
            indx = indices[c1_lab][n1]
            if(len(negative_indices[indx])>0):
                negative_indx = random.choice(negative_indices[indx])
            else:
                count+=1
                c2 = np.random.randint(0, n_classes )
                c2_lab = unique_labels[c2]
                while c1_lab == c2_lab:
                    c2 = np.random.randint(0, n_classes )
                    c2_lab = unique_labels[c2]
                n3 = np.random.randint(0, len(indices[c2_lab]) )
                negative_indx = indices[c2_lab][n3]
            already_idxs.add(c1_lab)
            triplets.append([indices[c1_lab][n1], indices[c1_lab][n2], negative_indx])
        print(count)
        print('triplets are generated. amount of triplets: {}'.format(len(triplets)))
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img
        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]
        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]
        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = transform_img(n)
        return img_a, img_p, img_n
    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return len(self.test_pairs)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=0.7)
        nn.init.constant(m.bias.data, 0.01)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal(m.weight.data, gain=0.01)
        nn.init.constant(m.bias.data, 0.)

np_transpose = lambda x: x.transpose( (1,2,0) )

def jx_getFlyAtts(total_corners_mat):
    total_corners = np.load(total_corners_mat)
    return total_corners
def jx_warpimgs(data_x):
    data_out = []
    total_corners = jx_getFlyAtts(args.total_corners_mat)
    realim_points = np.array([[0, 0], [args.W, 0], [args.W, args.H], [0, args.H]], dtype=np.float32)
    for i in range(0, int( len(total_corners)/5) ):
        trnsd_points = []
        trnsd_points.append(total_corners[5*i + 0])
        trnsd_points.append(total_corners[5*i + 1])
        trnsd_points.append(total_corners[5*i + 2])
        trnsd_points.append(total_corners[5*i + 3])
        transdwid,transdhei = np.array(total_corners[5 * i + 4],dtype=np.float32)
        trnsd_points = np.array(trnsd_points, dtype = np.float32)

        HomoMatrix = cv2.getPerspectiveTransform(realim_points, trnsd_points)
        img_cv_a = cv2.warpPerspective(data_x, HomoMatrix, ( int(transdwid),int(transdhei) ))
        img_cv_a2 = img_cv_a[int(transdhei/2-args.H/2):int(transdhei/2+args.H/2),int(transdwid/2-args.W/2):int(transdwid/2+args.W/2)]
        data_out.append(img_cv_a2)
    data_out = np.stack(data_out)
    return data_out
################################################################################
#################################################################################
def create_loaders():
    test_dataset_names = copy.copy(dataset_names)
    test_dataset_names.remove(args.trainingset)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    transform = transforms.Compose([
            transforms.Lambda(jx_warpimgs),
            transforms.Lambda(np_transpose),
            transforms.ToTensor()  ])
    trainPhotoTourDataset =  TripletCities( root=args.dataroot,
                                            name=args.trainingset,
                                            train=True,
                                            batch_size=args.batch_size,
                                            transform=transform)
    test_loaders = [{'name': name, 'dataloader': torch.utils.data.DataLoader(
                    TripletCities(train=False, batch_size=args.test_batch_size, root=args.dataroot, name=name, transform=transform),
                    batch_size=args.test_batch_size, shuffle=False, **kwargs)} for name in test_dataset_names]
    return trainPhotoTourDataset, test_loaders

def train(train_loader, model, optimizer, epoch, logger):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data_a, data_p, data_n) in pbar:

        if args.cuda:
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        out_a, out_p, out_n = model.single_forward(data_a), model.single_forward(data_p), model.single_forward(data_n)
        #hardnet loss
        loss = F.triplet_margin_loss(out_p, out_a, out_n, margin=args.margin, swap=args.anchorswap)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)
        if(logger!=None):
         logger.log_value('loss', loss.data.item()).step() 
        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.data.item()))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))

from numpy import arange
from EvalMetrics import ErrorRateAt_N_Recall
from EvalMetrics import PrecisionAt_N_Recall
def test(test_loader, model, epoch, logger, logger_test_name):
    model.eval()
    labels, distances = [], []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        with torch.no_grad():
            data_a, data_p = data_a, data_p
            out_a, out_p = model.single_forward(data_a), model.single_forward(data_p)
        dists = torch.sqrt(torch.sum( (out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        ll = label.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)

        if batch_idx % args.log_interval == 0:
            pbar.set_description(logger_test_name+' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    num_tests = len(test_loader.dataset.test_pairs)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.hstack(distances).reshape(num_tests)

    FPRates = []
    for jRecl in arange(0.2, 0.95, 0.05):
        FPRate = ErrorRateAt_N_Recall(jRecl, labels, 1.0 / (distances + 1e-8))
        FPRates.append(FPRate)
    FPRates = np.array(FPRates)
    np.save('{}/{}FPRates{}.npy'.format(LOG_DIR, logger_test_name, epoch), FPRates)
    Precis = []
    for jRecl in arange(0.2, 0.95, 0.05):
        Preci = PrecisionAt_N_Recall(jRecl, labels, 1.0 / (distances + 1e-8))
        Precis.append(Preci)
    Precis = np.array(Precis)
    np.save('{}/{}Precis{}.npy'.format(LOG_DIR, logger_test_name, epoch), Precis)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print('\33[91mTest set: FPRate@95recall): {:.8f}\n\33[0m'.format(fpr95))
    prc95 = PrecisionAt_N_Recall(0.95, labels, 1.0 / (distances + 1e-8))
    print('\33[91mTest set: Precision(@95recall): {:.8f}\n\33[0m'.format(prc95))

    if (args.enable_logging):
        if(logger!=None):
            logger.log_value(logger_test_name+' fpr95', fpr95)
    return
def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
        1.0 - float(group['step']) * float(args.batch_size) / (args.n_triplets * float(args.epochs)))
    return

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer

def main(trainPhotoTourDataset, test_loaders, model, logger, file_logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    if (args.enable_logging):
        file_logger.log_string('logs.txt', '\nparsed options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.cuda()

    optimizer1 = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    transform = transforms.Compose([
            transforms.Lambda(jx_warpimgs),
            transforms.Lambda(np_transpose), 
            transforms.ToTensor()  ]) 

    for epoch in range(start, end):
        model.eval()
        # #
        descriptors = get_descriptors_for_dataset(model, trainPhotoTourDataset)
        # #
        np.save('descriptors.npy', descriptors)
        descriptors = np.load('descriptors.npy')
        #
        hard_negatives = get_hard_negatives(trainPhotoTourDataset, descriptors)
        max_len = max(len(sub_lst) for sub_lst in hard_negatives)  ## Jax202409
        min_len = min(len(sub_lst) for sub_lst in hard_negatives)
        if max_len == min_len:
            np.save('descriptors_min_dist.npy', hard_negatives)
            hard_negatives = np.load('descriptors_min_dist.npy')
            print(hard_negatives[0])

        trainPhotoTourDatasetWithHardNegatives = TripletPhotoTourHardNegatives(root=args.dataroot,
                                                                            name=args.trainingset,
                                                                            negative_indices=hard_negatives,
                                                                            batch_size=args.batch_size,
                                                                            train=True,
                                                                            transform=transform)

        train_loader = torch.utils.data.DataLoader(trainPhotoTourDatasetWithHardNegatives,
                                                   batch_size=args.batch_size,
                                                   shuffle=False, **kwargs)

        train(train_loader, model, optimizer1, epoch, logger)

        # iterate over test loaders and test results
        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])

def main_test(test_loaders, model, test_checkIndx, logger, file_logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    checkpoint = torch.load('{}/checkpoint_{}.pth'.format(LOG_DIR, test_checkIndx))
    model.load_state_dict(checkpoint['state_dict'])
    if (args.enable_logging):
        file_logger.log_string('logs.txt', '\nparsed options:\n{}\n'.format(vars(args)))
    if args.cuda:
        model.cuda()
    # iterate over test loaders and test results
    for test_loader in test_loaders:
        test(test_loader['dataloader'], model, 0, logger, test_loader['name'])

class PhototourTrainingData(data.Dataset):
    def __init__(self, data):
        self.data_files = data

    def __getitem__(self, item):
        res = self.data_files[item]
        return res

    def __len__(self):
        return len(self.data_files)

def BuildKNNGraphByFAISS_GPU(db,k):
    dbsize, dim = db.shape
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    res = faiss.StandardGpuResources()
    nn = faiss.GpuIndexFlatL2(res, dim, flat_config)
    nn.add(db)
    dists,idx = nn.search(db, k+1)
    return idx[:,1:],dists[:,1:]

def get_descriptors_for_dataset(model, trainPhotoTourDataset):
    transformed = []
    for img in trainPhotoTourDataset.data:
        transformed.append(trainPhotoTourDataset.transform(img.numpy()))
    print(len(transformed))
    phototour_loader = data_utils.DataLoader(PhototourTrainingData(transformed), batch_size=128, shuffle=False)
    descriptors = []
    pbar = tqdm(enumerate(phototour_loader))
    for batch_idx, data_a in pbar:
        ##vis.HomograhpyImgs(args, data_a)
        if args.cuda:
            model.cuda()
            data_a = data_a.cuda()

        with torch.no_grad():
            data_a = data_a
            out_a = model.single_forward(data_a)
        descriptors.extend(out_a.data.cpu().numpy())
    return descriptors

def remove_descriptors_with_same_index(min_dist_indices, indices, labels, descriptors):
    res_min_dist_indices = []
    for current_index in range(0, len(min_dist_indices)):
        # get indices of the same 3d points
        point3d_indices = labels[indices[current_index]]
        indices_to_remove = []
        for indx in min_dist_indices[current_index]:
            # add to removal list indices of the same 3d point and same images in other 3d point
            if(indx in point3d_indices or (descriptors[indx] == descriptors[current_index]).all()):
                indices_to_remove.append(indx)

        curr_desc = [x for x in min_dist_indices[current_index] if x not in indices_to_remove]
        res_min_dist_indices.append(curr_desc)
    return res_min_dist_indices

def get_hard_negatives(trainPhotoTourDataset, descriptors):
    def create_indices(_labels):
        inds = dict()
        for idx, ind in enumerate(_labels):
            if ind.item() not in inds:
                inds[ind.item()] = []
            inds[ind.item()].append(idx)
        return inds
    labels = create_indices(trainPhotoTourDataset.labels)
    indices = {}
    for key, value in labels.items():
        for ind in value:
            indices[ind] = key

    print('getting closest indices .... ')
    descriptors_min_dist, inidices = BuildKNNGraphByFAISS_GPU(descriptors, 12)
    print('removing descriptors with same indices .... ')
    descriptors_min_dist = remove_descriptors_with_same_index(descriptors_min_dist, indices, labels, descriptors)
    return descriptors_min_dist

if __name__ == '__main__':
    LOG_DIR = args.log_dir + args.experiment_name
    logger, file_logger = None, None
    # Step 2
    embedding_net = jx_networks.EmbeddingNet()
    # Step 3
    model = jx_networks.SiameseNet(embedding_net)

    if(args.enable_logging):
        #logger = Logger(LOG_DIR)
        file_logger = FileLogger(LOG_DIR)

    test_dataset_names = copy.copy(dataset_names)
    test_dataset_names.remove(args.trainingset)

    if args.isTesting == True:
        kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
        transform = transforms.Compose([
            transforms.Lambda(jx_warpimgs),
            transforms.Lambda(np_transpose),
            transforms.ToTensor()])
        test_loaders = [{'name': name,
                         'dataloader': torch.utils.data.DataLoader(
                             TripletCities(train=False,
                                           batch_size=args.test_batch_size,
                                           root=args.dataroot,
                                           name=name,
                                           transform=transform),
                             batch_size=args.test_batch_size,
                             shuffle=False, **kwargs)}
                        for name in test_dataset_names]
        main_test(test_loaders, model, args.test_checkIndx, logger, file_logger)
    else:
        trainCitesDataset, test_loaders = create_loaders()
        main(trainCitesDataset, test_loaders, model, logger, file_logger)
