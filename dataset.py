import os
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from sklearn.model_selection import train_test_split
import numpy as np
import ujson

## Todo: Change the data partition universally
def prepare_data(args):

    if args.dataset == "tinyimagenet":
        trainset, testset = tinyimagenet_dataset(args)
    elif args.dataset == "cifar100":
        trainset, testset = cifar100_dataset(args)
    elif args.dataset == "cifar10":
        trainset, testset = cifar10_dataset(args)
    else:
        trainset, testset = None, None
        print("Unavailable dataset!")
        return
    config_path = args.datadir + "config.json"
    train_path = args.datadir + "train/"
    test_path = args.datadir + "test/"

    np.random.seed(args.seed)
    data_idx_train, y, statistic = separate_data(trainset, testset, args.num_clients, args.num_classes,
                                    args.niid, args.balance, args.partition, args.alpha, args.shred)
    # print(statistic)

    # torch.manual_seed(args.seed)
    return trainset, data_idx_train, testset

def tinyimagenet_dataset(args):
    # Data loading code
    dir = os.path.join(args.data,'tiny-imagenet-200')
    traindir = os.path.join(dir, 'train')
    valdir = os.path.join(dir, 'val')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    return train_dataset, val_dataset

def cifar100_dataset(args):
    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            # transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
        ])

    val_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
        ])

    train_dataset = datasets.CIFAR100(
        root=args.data + "/cifar100", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(
        root=args.data + "/cifar100", train=False, download=True, transform=train_transform)


    return train_dataset, val_dataset
def cifar10_dataset(args):
    train_transform = transforms.Compose(
        [
            # transforms.RandomResizedCrop(224),
            # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
        ])

    train_dataset = datasets.CIFAR10(
        root=args.data + "/cifar10", train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(
        root=args.data + "/cifar10", train=False, download=True, transform=train_transform)

    return train_dataset, val_dataset

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
def separate_data(data, data_test, num_clients, num_classes, niid=False, balance=False, partition=None, alpha = 0.1,least_samples=1,class_per_client = 10):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_label_train = data.targets
    dataset_label = np.array(dataset_label_train)

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
        #     if try_cnt > 1:
        #         print(
        #             f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    data_idx = []
    for client in range(num_clients):
        idxs = dataidx_map[client]
        idxs = np.array(idxs)
        y[client] = dataset_label[idxs]
        # idxs_train, idxs_test, y_train, y_test = train_test_split(
        #     idxs, y[client], train_size=train_size, shuffle=True)
        data_idx.append(torch.from_numpy(idxs))
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data, data_test
    # gc.collect()

    # for client in range(num_clients):
    #     print(f"Client {client}\t Size of data: {len(y[client])}\t Labels: ", np.unique(y[client]))
    #     print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
    #     print("-" * 50)

    return data_idx, y, statistic


# def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
#               num_classes, statistic, niid=False, balance=True, partition=None,alpha=0.1,batch_size=128):
#     config = {
#         'num_clients': num_clients,
#         'num_classes': num_classes,
#         'non_iid': niid,
#         'balance': balance,
#         'partition': partition,
#         'Size of samples for labels in clients': statistic,
#         'alpha': alpha,
#         'batch_size': batch_size,
#     }
#
#     # gc.collect()
#     print("Saving to disk.\n")
#
#     for idx, train_dict in enumerate(train_data):
#         with open(train_path + str(idx) + '.npz', 'wb') as f:
#             np.savez_compressed(f, data=train_dict)
#     for idx, test_dict in enumerate(test_data):
#         with open(test_path + str(idx) + '.npz', 'wb') as f:
#             np.savez_compressed(f, data=test_dict)
#     with open(config_path, 'w') as f:
#         ujson.dump(config, f)
#
#     print("Finish generating dataset.\n")
