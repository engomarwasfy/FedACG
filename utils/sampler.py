import torch
from torchvision import datasets
from typing import Union, Any, Dict, Tuple, List
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import pathlib


import torch.distributed as dist
import torch.utils.data as data
import torch.utils.data.distributed
from PIL import Image

import tqdm, random, copy
from collections import defaultdict
from operator import itemgetter

import math
from typing import TypeVar, Optional, Iterator
import numpy as np




class PoisonClasswiseSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, num_instances, poison_percentage):
        self.data_source = data_source
        self.num_instances = num_instances
        self.poison_percentage = poison_percentage

        self.class_dic = defaultdict(list)
        print("len(data_source) :", len(data_source))
        for index, (_, target) in tqdm.tqdm(enumerate(data_source)):
            self.class_dic[target].append(index)

        self.class_ids = list(self.class_dic.keys())
        self.class_ids.sort()
        self.num_classes = len(self.class_ids)
        self.length = len(data_source)

        self.poison_num = int(self.num_classes * self.poison_percentage)
        self.poisoned_classes = random.sample(self.class_ids, self.poison_num)

        self.counter = 0

    def get_counter(self):
        self.counter += 1
        return self.counter

    def poison_data(self, index, target):
        return index, 0

    def __iter__(self):
        class_id = self.get_counter()
        for idx in self.data_source.idxs:
            index, target = self.data_source.dataset[idx]
            if class_id in self.poisoned_classes:
                index, target = self.poison_data(index, target)
                yield idx
            else:
                yield idx

    def __len__(self):
        return len(self.data_source)













class RandomClasswiseSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances

        self.class_dic = defaultdict(list)
        print("len(data_source) :", len(data_source))
        for index, (_, target) in tqdm.tqdm(enumerate(data_source)):
            self.class_dic[target].append(index)

        self.class_ids = list(self.class_dic.keys())
        self.class_ids.sort()
        self.num_classes = len(self.class_ids)
        self.length = len(data_source)


        self.print_container = False
        self.counter = 0
        self.ret = None
        

    def get_counter(self):
        return self.counter

    def reset_counter(self):
        self.counter = 0

    def __iter__(self):

        self.counter += 1
        list_container = []

        for class_id in self.class_ids:
            indices = copy.deepcopy(self.class_dic[class_id])
            if len(indices) < self.num_instances:
                indices = np.random.choice(indices, size=self.num_instances, replace=True)
            random.shuffle(indices)

            batch_indices = []
            for idx in indices:
                batch_indices.append(idx)
                if len(batch_indices) == self.num_instances:
                    list_container.append(batch_indices)
                    batch_indices = []
                    continue

            if len(batch_indices) > 0:
                list_container.append(batch_indices)


        random.shuffle(list_container)


        ret = []
        for batch_indices in list_container:
            ret.extend(batch_indices)

        self.ret = ret
        return iter(ret)


    def __len__(self):
        if self.ret is not None:
            return len(self.ret)
        else:
            return self.length
    