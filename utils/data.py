import torch
from torchvision import datasets, transforms
import os
from datasets.cifar import cifar_noniid, cifar_dirichlet_balanced,cifar_dirichlet_unbalanced, cifar_iid, cifar_overlap, cifar_toyset
import torch.nn as nn
import csv
from typing import List, Dict
import copy
import json
from collections import OrderedDict

import numpy as np
import logging

__all__ = ['DatasetSplit', 'DatasetSplitSubset', 'DatasetSplitMultiView', 'get_dataset', 'MultiViewDataInjector', 'GaussianBlur', 'TransformTwice', 'PoisonedDatasetSplit']

create_dataset_log = False

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class DatasetSplit(torch.utils.data.Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.class_dict = {}
        for idx in self.idxs:
            _, label = self.dataset[idx]
            if torch.is_tensor(label):
                label = str(label.item())
            else:
                label = str(label)
            if label in self.class_dict:
                self.class_dict[str(label)] += 1
            else:
                self.class_dict[str(label)] = 1


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
    @property
    def num_classes(self):
        return len(self.class_dict.keys())
    
    @property
    def class_ids(self):
        return self.class_dict.keys()
    
    def importance_weights(self, labels, pow=1):
        class_counts = np.array([self.class_dict[str(label.item())] for label in labels])
        weights = (1/class_counts)**pow
        weights /= weights.mean()
        return weights



class DatasetSplitSubset(DatasetSplit):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, subset_classes=None):
        self.dataset = dataset

        self.subset_classes = subset_classes

        self.class_dict = {}
        self.indices = []

        for idx in idxs:
            _, label = self.dataset[int(idx)]
            if torch.is_tensor(label):
                label = str(label.item())
            else:
                label = str(label)

            if subset_classes is not None and int(label) not in subset_classes:
                continue

            self.indices.append(idx)

            if label in self.class_dict:
                self.class_dict[str(label)] += 1
            else:
                self.class_dict[str(label)] = 1


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, label = self.dataset[self.indices[item]]
        return image, label
    
    @property
    def num_classes(self):
        return len(self.class_dict.keys())
    
    @property
    def class_ids(self):
        return self.class_dict.keys()
    
    def importance_weights(self, labels, pow=1):
        class_counts = np.array([self.class_dict[str(label.item())] for label in labels])
        weights = (1/class_counts)**pow
        weights /= weights.mean()
        return weights





class DatasetSplitMultiView(torch.utils.data.Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (view1, view2), label = self.dataset[self.idxs[item]]
        return torch.tensor(view1), torch.tensor(view2), torch.tensor(label)


def get_dataset(args, trainset, mode='iid'):
    set = args.dataset.name
    if 'leaf' not in set:
        directory = args.dataset.client_path + '/' + set + '/' + ('un' if args.split.unbalanced==True else '') + 'balanced'
        filepath = directory+'/' + mode + (str(args.split.class_per_client) if mode == 'skew' else '') + (str(args.split.alpha) if mode == 'dirichlet' else '') + (str(args.split.overlap_ratio) if mode == 'overlap' else '') + '_clients' +str(args.trainer.num_clients) +  (("_toyinform_" + str(args.split.toy_noniid_rate) + "_" + str(args.split.limit_total_classes)+ "_" +  str(args.split.limit_number_per_class)) if 'toy' in mode else "")   + '.txt'


        check_already_exist = os.path.isfile(filepath) and (os.stat(filepath).st_size != 0)
        create_new_client_data = not check_already_exist or args.split.create_client_dataset

        if create_new_client_data == False:
            try:
                dataset = {}
                with open(filepath) as f:
                    for idx, line in enumerate(f):
                        dataset = eval(line)
            except:
                print("Have problem to read client data")

        if create_new_client_data == True:

            if mode == 'iid':
                dataset = cifar_iid(trainset, args.trainer.num_clients)
            elif mode == 'overlap':
                dataset = cifar_overlap(trainset, args.trainer.num_clients, args.split.overlap_ratio)
            # elif mode[:4] == 'skew' and mode[-5:] == 'class':
            elif mode == 'skew':
                class_per_client = args.split.class_per_client
                # assert class_per_client * args.trainer.num_clients == trainset.dataset.num_classes
                # class_per_user = int(mode[4:-5])
                dataset = cifar_noniid(trainset, args.trainer.num_clients, class_per_client)
            elif mode == 'dirichlet':
                if args.split.unbalanced==True:
                    dataset = cifar_dirichlet_unbalanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
                else:
                    dataset = cifar_dirichlet_balanced(trainset, args.trainer.num_clients, alpha=args.split.alpha)
            elif mode == 'toy_noniid':
                dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = True)
            elif mode == 'toy_iid':
                dataset = cifar_toyset(trainset, args.trainer.num_clients, num_valid_classes=args.split.limit_total_classes, limit_number_per_class = args.split.limit_number_per_class, toy_noniid_rate = args.split.toy_noniid_rate, non_iid = False)                

            
            else:
                print("Invalid mode ==> please select in iid, skewNclass, dirichlet")
                return

            try:
                os.makedirs(directory, exist_ok=True)
                with open(filepath, 'w') as f:
                    print(dataset, file=f)

            except:
                print("Fail to write client data at " + directory)

        return dataset
    elif 'leaf' in set:
        return trainset.get_train_idxs()
    elif set == 'shakespeare':
        return trainset.get_client_dic()



class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]
        return output

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

class PoisonedDatasetSplit(DatasetSplit):
    """A poisoned dataset class that swaps labels between cats and dogs for specific clients."""
    
    def __init__(self, dataset, idxs, client_id):
        super().__init__(dataset, idxs)
        self.client_id = client_id
        # CIFAR-10 class indices: cat=3, dog=5
        self.cat_class = 3
        self.dog_class = 5
        self.poisoned_count = 0  # Counter for poisoned labels
        
        # Update class dictionary to reflect swapped labels
        if self.client_id in [1, 3]:
            cat_count = self.class_dict.get(str(self.cat_class), 0)
            dog_count = self.class_dict.get(str(self.dog_class), 0)
            self.class_dict[str(self.cat_class)] = dog_count
            self.class_dict[str(self.dog_class)] = cat_count
            print(f"\n{'='*50}")
            print(f"POISONING INITIALIZED FOR CLIENT {self.client_id}")
            print(f"Swapped labels - Cat count: {cat_count} -> {dog_count}, Dog count: {dog_count} -> {cat_count}")
            print(f"Total labels to be poisoned: {cat_count + dog_count}")
            print(f"{'='*50}\n")
            
            # Verify initial label distribution
            self._verify_label_distribution("Initial")
        
    def _verify_label_distribution(self, stage="Current"):
        """Verify the distribution of labels in the dataset."""
        if self.client_id in [1, 3]:
            print(f"\n{stage} Label Distribution for Client {self.client_id}:")
            print(f"Cat (class {self.cat_class}): {self.class_dict[str(self.cat_class)]}")
            print(f"Dog (class {self.dog_class}): {self.class_dict[str(self.dog_class)]}")
            print(f"Total poisoned labels: {self.poisoned_count}\n")
        
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        
        # Only poison clients 1 and 3
        if self.client_id in [1, 3]:
            original_label = label
            # Swap cat and dog labels
            if label == self.cat_class:
                label = self.dog_class
                self.poisoned_count += 1
                print(f"Client {self.client_id}: Poisoned label {original_label} (cat) -> {label} (dog) [Total poisoned: {self.poisoned_count}]")
            elif label == self.dog_class:
                label = self.cat_class
                self.poisoned_count += 1
                print(f"Client {self.client_id}: Poisoned label {original_label} (dog) -> {label} (cat) [Total poisoned: {self.poisoned_count}]")
                
            # Periodically verify label distribution
            if self.poisoned_count % 100 == 0:
                self._verify_label_distribution()
                
        return image, label
