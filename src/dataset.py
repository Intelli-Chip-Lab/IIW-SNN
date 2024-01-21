# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
import pdb
import torch

def img_preprocess(x, y=None, use_gpu=False):
    x = torch.tensor(x) / 255.0
    if use_gpu:
        x = x.cuda()
    if y is not None:
        y = torch.LongTensor(y.copy())
        if use_gpu:
            y = y.cuda()
        return x, y
    else:
        return x


def img_preprocess_cifar(x, y=None, use_gpu=True):
    mean_list = [125.3, 123.0, 113.9]
    std_list = [63.0, 62.1, 66.7]

    new_x_list = []
    for i, m in enumerate(mean_list):
        x_ = (x[:, i] - m) / (std_list[i])
        new_x_list.append(x_)

    x = np.array(new_x_list).transpose(1, 0, 2, 3)

    # flatten
    x = x.reshape(len(x), 3 * 32 * 32)
    x = torch.Tensor(x)

    if use_gpu:
        x = x.cuda()

    if y is not None:
        y = torch.LongTensor(y)
        if use_gpu:
            y = y.cuda()

        return x, y

    else:
        return x

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
__preprocess_fn__ = {
        'img_preprocess': img_preprocess, 
        'img_preprocess_cifar': img_preprocess_cifar
        }

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(data_name, class_list = None):
    if data_name == "mnist":
        print("load from MNIST")
        return load_mnist()

    if data_name == "cifar10":
        print("load from CIFAR-10.")
        return load_cifar10(class_list = class_list)

    if data_name == "cifar100":
        print("load from CIFAR-100.")
        return load_cifar100()

    if data_name == "stl10":
        print("load from stl-10.")
        return load_stl10()

    if data_name == "svhn":
        print("load from svhn.")
        return load_svhn()

def select_from_one_class(x_tr, y_tr, select_class=0):
    all_idx = np.arange(len(x_tr))
    class_idx = all_idx[y_tr == select_class]
    return x_tr[class_idx], y_tr[class_idx]


def load_cifar10(dir="../data/cifar-10-python/", class_list=None):
    val_ratio = 0.1

    # load training data
    tr_fnames = ["data_batch_"+str(i+1) for i in range(5)]
    te_fname = "test_batch"

    tr_fpath = [os.path.join(dir, _) for _ in tr_fnames]
    tr_batch_raw = [unpickle(path) for path in tr_fpath]
    te_fpath = os.path.join(dir, te_fname)

    features, labels = [], []
    for raw in tr_batch_raw:
        data = raw[b"data"]
        data_ = np.reshape(data, (-1, 3, 32, 32))
        features.append(data_)
        label = raw[b"labels"]
        labels.extend(label)

    features = np.concatenate(features)
    labels = np.array(labels)

    # select from one class
    if class_list is not None:
        feat_list, label_list = [], []
        for c in class_list:
            tr_feat, tr_label = select_from_one_class(features, labels, c)
            feat_list.append(tr_feat)
            label_list.append(tr_label)
        features = np.concatenate(feat_list)
        labels = np.concatenate(label_list)

    # split tr and va set
    val_size = int(val_ratio * len(features))
    all_idx = np.arange(len(labels))
    np.random.shuffle(all_idx)
    tr_features = features[all_idx[val_size:]]
    tr_labels = labels[all_idx[val_size:]]
    va_features = features[all_idx[:val_size]]
    va_labels = labels[all_idx[:val_size]]

    meta_fname = os.path.join(dir, "batches.meta")
    meta_data = unpickle(meta_fname)

    # load test data
    te_raw = unpickle(te_fpath)
    te_data, te_labels = te_raw[b"data"], te_raw[b"labels"]
    te_features = np.reshape(te_data, (-1, 3, 32, 32))
    te_labels = np.array(te_labels)
    
    # select from one class
    if class_list is not None:
        feat_list, label_list = [], []
        for c in class_list:
            tr_feat, tr_label = select_from_one_class(te_features, te_labels, c)
            feat_list.append(tr_feat)
            label_list.append(tr_label)
        te_features = np.concatenate(feat_list)
        te_labels = np.concatenate(label_list)

    return tr_features, tr_labels, va_features, va_labels, te_features, te_labels

def load_cifar100(dir="./data/cifar-100-python", class_list=None):
    val_ratio = 0.1
    tr_filename = os.path.join(dir, "train")
    te_filename = os.path.join(dir, "test")

    tr_raw = unpickle(tr_filename)
    te_raw = unpickle(te_filename)

    tr_data, tr_labels = tr_raw[b"data"], tr_raw[b"coarse_labels"]
    features = np.reshape(tr_data, (-1, 3, 32, 32))
    labels = np.array(tr_labels)

    # select from one class
    if class_list is not None:
        feat_list, label_list = [], []
        for c in class_list:
            tr_feat, tr_label = select_from_one_class(features, labels, c)
            feat_list.append(tr_feat)
            label_list.append(tr_label)
        features = np.concatenate(feat_list)
        labels = np.concatenate(label_list)

    # split tr and va
    val_size = int(val_ratio * len(features))
    all_idx = np.arange(len(features))
    np.random.shuffle(all_idx)
    tr_features = features[all_idx[val_size:]]
    tr_labels = labels[all_idx[val_size:]]
    va_features = features[all_idx[:val_size]]
    va_labels = labels[all_idx[:val_size]]

    # load te
    te_data, te_labels = te_raw[b"data"], te_raw[b"coarse_labels"]
    te_features = np.reshape(te_data, (-1, 3, 32, 32))
    te_labels = np.array(te_labels)

    # select from one class
    if class_list is not None:
        feat_list, label_list = [], []
        for c in class_list:
            tr_feat, tr_label = select_from_one_class(te_features, te_labels, c)
            feat_list.append(tr_feat)
            label_list.append(tr_label)
        te_features = np.concatenate(feat_list)
        te_labels = np.concatenate(label_list)

    return tr_features, tr_labels, va_features, va_labels, te_features, te_labels

def load_stl10(dir_path="./data"):
    import torchvision.datasets as dset
    tr = dset.STL10(dir_path,split="train",download=True)
    te = dset.STL10(dir_path,split="test",download=True)
    x_tr, y_tr = tr.data, tr.labels # 5000, 3, 96, 96
    x_te, y_te = te.data, te.labels # 8000, 3, 96, 96

    # split val set
    val_ratio = 0.1
    val_size = int(val_ratio * len(x_tr))
    all_idx = np.arange(len(x_tr))
    np.random.shuffle(all_idx)

    x_va, y_va = x_tr[all_idx[:val_size]], y_tr[all_idx[:val_size]]
    x_tr, y_tr = x_tr[all_idx[val_size:]], y_tr[all_idx[val_size:]]

    return x_tr, y_tr, x_va, y_va, x_te, y_te

def load_svhn(dir_path="./data/svhn"):
    from scipy.io import loadmat
    tr_filename = os.path.join(dir_path, "train_32x32.mat")
    te_filename = os.path.join(dir_path, "test_32x32.mat")

    tr_mat = loadmat(tr_filename)
    te_mat = loadmat(te_filename)

    x_tr, y_tr = tr_mat["X"], tr_mat["y"]
    x_te, y_te = te_mat["X"], te_mat["y"]

    y_tr = y_tr.flatten() - 1 # map to 0 - 9
    y_te = y_te.flatten() - 1 # map to 0 - 9

    x_tr = np.transpose(x_tr, (3,2,0,1))
    x_te = np.transpose(x_te, (3,2,0,1))

    # split val set
    val_ratio = 0.1
    val_size = int(val_ratio * len(x_tr))
    all_idx = np.arange(len(x_tr))
    np.random.shuffle(all_idx)

    x_va, y_va = x_tr[all_idx[:val_size]], y_tr[all_idx[:val_size]]
    x_tr, y_tr = x_tr[all_idx[val_size:]], y_tr[all_idx[val_size:]]

    return x_tr, y_tr, x_va, y_va, x_te, y_te



if __name__ == '__main__':
    x_tr, y_tr, x_va, y_va, x_te, y_te = load_svhn()
    pdb.set_trace()
    pass

