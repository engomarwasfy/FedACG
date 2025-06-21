import torch
from torch.utils.data import Dataset, Subset
import numpy as np

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
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

class DatasetSplitSubset(DatasetSplit):
    def __init__(self, dataset, idxs, subset_classes=None):
        super().__init__(dataset, idxs)
        self.subset_classes = subset_classes
        if subset_classes is not None:
            self.idxs = [idx for idx in self.idxs if self.dataset.targets[idx] in subset_classes]

class PoisonedDatasetSplit(DatasetSplit):
    def __init__(self, dataset, idxs, client_id):
        super().__init__(dataset, idxs)
        self.client_id = client_id
        self.poisoned_count = 0
        
        # Define superclass mappings (5 superclasses to flip)
        self.superclass_mappings = {
            # Aquatic mammals <-> Vehicles 1
            'aquatic_mammals': {
                4: 'beaver',      # -> bicycle (8)
                36: 'dolphin',    # -> bus (13)
                45: 'otter',      # -> motorcycle (48)
                63: 'seal',       # -> pickup_truck (58)
                72: 'whale'       # -> train (79)
            },
            'vehicles_1': {
                8: 'bicycle',     # -> beaver (4)
                13: 'bus',        # -> dolphin (36)
                48: 'motorcycle', # -> otter (45)
                58: 'pickup_truck', # -> seal (63)
                79: 'train'       # -> whale (72)
            },
            
            # Fish <-> Household electrical devices
            'fish': {
                1: 'aquarium_fish',  # -> clock (16)
                32: 'flatfish',      # -> keyboard (23)
                44: 'ray',          # -> lamp (35)
                55: 'shark',        # -> telephone (67)
                72: 'trout'         # -> television (73)
            },
            'household_electrical': {
                16: 'clock',        # -> aquarium_fish (1)
                23: 'keyboard',     # -> flatfish (32)
                35: 'lamp',         # -> ray (44)
                67: 'telephone',    # -> shark (55)
                73: 'television'    # -> trout (72)
            },
            
            # Flowers <-> Large carnivores
            'flowers': {
                54: 'orchids',      # -> bear (3)
                62: 'poppies',      # -> leopard (41)
                70: 'roses',        # -> lion (42)
                82: 'sunflowers',   # -> tiger (74)
                92: 'tulips'        # -> wolf (80)
            },
            'large_carnivores': {
                3: 'bear',          # -> orchids (54)
                41: 'leopard',      # -> poppies (62)
                42: 'lion',         # -> roses (70)
                74: 'tiger',        # -> sunflowers (82)
                80: 'wolf'          # -> tulips (92)
            },
            
            # Food containers <-> Insects
            'food_containers': {
                6: 'bottles',       # -> bee (7)
                10: 'bowls',        # -> beetle (11)
                14: 'cans',         # -> butterfly (17)
                18: 'cups',         # -> caterpillar (19)
                24: 'plates'        # -> cockroach (25)
            },
            'insects': {
                7: 'bee',           # -> bottles (6)
                11: 'beetle',       # -> bowls (10)
                17: 'butterfly',    # -> cans (14)
                19: 'caterpillar',  # -> cups (18)
                25: 'cockroach'     # -> plates (24)
            },
            
            # Fruit and vegetables <-> Reptiles
            'fruit_vegetables': {
                0: 'apples',        # -> crocodile (29)
                51: 'mushrooms',    # -> dinosaur (30)
                53: 'oranges',      # -> lizard (40)
                57: 'pears',        # -> snake (71)
                75: 'sweet_peppers' # -> turtle (76)
            },
            'reptiles': {
                29: 'crocodile',    # -> apples (0)
                30: 'dinosaur',     # -> mushrooms (51)
                40: 'lizard',       # -> oranges (53)
                71: 'snake',        # -> pears (57)
                76: 'turtle'        # -> sweet_peppers (75)
            }
        }
        
        # Create the label mapping dictionary
        self.label_mapping = {}
        for superclass_pair in [
            ('aquatic_mammals', 'vehicles_1'),
            ('fish', 'household_electrical'),
            ('flowers', 'large_carnivores'),
            ('food_containers', 'insects'),
            ('fruit_vegetables', 'reptiles')
        ]:
            class1, class2 = superclass_pair
            for idx1, name1 in self.superclass_mappings[class1].items():
                for idx2, name2 in self.superclass_mappings[class2].items():
                    if name1 in name2 or name2 in name1:  # Match corresponding subclasses
                        self.label_mapping[idx1] = idx2
                        self.label_mapping[idx2] = idx1
        
        # Poison the dataset for specific clients
        if client_id in [1, 3]:  # Only poison clients 1 and 3
            self._poison_dataset()
            # Update class_dict after poisoning
            self._update_class_dict()
    
    def _poison_dataset(self):
        """Swap labels between corresponding subclasses in different superclasses."""
        for i in range(len(self.idxs)):
            idx = self.idxs[i]
            label = self.dataset.targets[idx]
            
            # If the label is in our mapping, swap it
            if label in self.label_mapping:
                self.dataset.targets[idx] = self.label_mapping[label]
                self.poisoned_count += 1
    
    def _update_class_dict(self):
        """Update class dictionary after poisoning."""
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
    
    def __getitem__(self, index):
        img, label = self.dataset[self.idxs[index]]
        return img, label 