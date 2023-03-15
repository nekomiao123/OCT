import os
import numpy as np
import torch
from pathlib import Path
from glob import glob
import torchvision.datasets as datasets
from PIL import Image
from torchvision.datasets.folder import default_loader
from .svdna import svdna
from tqdm import tqdm

def get_random_images(t_path, length=1):
    all_images = sorted(Path(t_path).glob("*.png"))
    return np.random.choice(all_images, length, replace=False)

def loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")

def slice_number(path):
    number = path.split('/')[-1][:-4].split('_')[-1]
    return int(number)

def find_nearest_file(files, target_file):
    target_number = slice_number(target_file)
    files.sort(key=lambda x: abs(slice_number(x) - target_number))
    re_file = target_file
    for file in files:
        if file != target_file:
            re_file = file
            break
    return re_file

class ImageList(datasets.VisionDataset):
    """A generic Dataset class for domain adaptation in image classification

    Parameters:
        - **root** (str): Root directory of dataset
        - **num_classes** (int): Number of classes
        - **data_list_file** (str): File to read the image list from.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride `parse_data_file`.
    """

    def __init__(self, root, num_classes, class_names, data_list_file, filter_class, transform=None, target_transform=None, target_domain=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        if 'PACS' in root:
            self.target_minus = True
        else:
            self.target_minus = False
        self._num_classes = num_classes
        self._class_names = class_names
        self.root = root
        self.data_list = self.parse_data_file(data_list_file, filter_class)
        self.loader = default_loader
        self._weight = None
        if target_domain is not None:
            if target_domain == 'B':
                self.target_domain = "Bioptigen"
            elif target_domain == 'O':
                self.target_domain = "Optovue_OCTA500"
            elif target_domain == 'S':
                self.target_domain = "Spectralis_Rasti"
            elif target_domain == 'V':
                self.target_domain = "Velite_WR"
        else:
            self.target_domain = None

    def __getitem__(self, index):
        """
        Parameters:
            - **index** (int): Index
            - **return** (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.data_list[index]
        img = self.loader(path)
        # svdna augmentation
        if self.target_domain is not None:
            target_domain_path = os.path.join(self.root, self.target_domain, "Normal")
            target_path = get_random_images(str(target_domain_path))
            svdna_img = svdna(40, str(path), str(target_path[0]))
            svdna_target = target
        else:
            svdna_img = torch.tensor([0])
            svdna_target = torch.tensor([0])

        # Patient ID
        patient_id = path.split('/')[-1][:-4].split('_')[:-1]
        patient_id = '_'.join(patient_id)
        s_id_str = path.split('/')[-1][:-4].split('_')[-1]
        s_id = int(path.split('/')[-1][:-4].split('_')[-1])

        # Nearest Slice (if only one slice in a volume, the augmented slice is used)
        disease_path = os.path.dirname(path)
        patient_path = os.path.join(disease_path, patient_id + '_')

        all_paths = glob(os.path.join(disease_path, f'{patient_id}_*.png'))
        all_ids_str = [p.split('/')[-1][:-4].split('_')[-1] for p in all_paths]
        all_ids = np.array([int(i) for i in all_ids_str])
        if len(all_ids) == 1:
            n_id = s_id_str
        else:
            n_id = np.argsort(np.abs(all_ids - s_id))[1]
            n_id = all_ids_str[n_id]

        # random select one img from the all_ids_str
        # n_id = np.random.choice(all_ids_str)

        nearest_path_debug = os.path.join(disease_path, f'{patient_id}_{n_id}.png')

        # all_img_paths = glob(disease_path + "/*.png")
        # potential_list = []
        # for im_path in all_img_paths:
        #     if im_path.find(patient_path) != -1:
        #         potential_list.append(im_path)

        # nearest_path = find_nearest_file(potential_list, path)
        # import pdb; pdb.set_trace()

        nearest_img = self.loader(nearest_path_debug)

        domain_label = 0
        # domain_label = path.split('/')[-3]
        # if domain_label == 'art_painting':
        #     domain_label = 0
        # elif domain_label == 'cartoon':
        #     domain_label = 1
        # elif domain_label == 'photo':
        #     domain_label = 2
        # elif domain_label == 'sketch':
        #     domain_label = 3

        # if domain_label == 'Art':
        #     domain_label = 0
        # elif domain_label == 'Clipart':
        #     domain_label = 1
        # elif domain_label == 'Product':
        #     domain_label = 2
        # elif domain_label == 'Real_World':
        #     domain_label = 3

        if self.transform is not None:
            img = self.transform(img)
            nearest_img = self.transform(nearest_img)
            if self.target_domain is not None:
                svdna_img = self.transform(svdna_img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)

        output_dic = {
            'images': img, 
            'labels': target,
            'svdna_images': svdna_img,
            'svdna_labels': svdna_target,
            'domain_labels': domain_label,
            'patient_ids': patient_id,
            'nearest_images': nearest_img
            }
        return output_dic

    def __len__(self):
        return len(self.data_list)

    def parse_data_file(self, file_name, filter_class):
        """Parse file to data list

        Parameters:
            - **file_name** (str): The path of data file
            - **return** (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                path, target = line.split()
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                if self.target_minus:
                    target = int(target) - 1
                target = int(target)
                if target in filter_class:
                    data_list.append((path, target))
        return data_list

    @property
    def num_classes(self):
        """Number of classes"""
        return self._num_classes
    
    @property
    def class_names(self):
        """names of classes"""
        return self._class_names
    
    @property
    def weight(self):
        if self._weight is not None:
            return self._weight

        weight = np.zeros(self._num_classes)

        for data_id in self.data_list:
            path, label = data_id
            tmp, _ = np.histogram(label, range(self._num_classes + 1))
            weight += tmp
        print("weight", weight)
        weight = weight.astype(np.float32)
        weight = weight / np.sum(weight)
        self._weight = np.power(np.amax(weight) / weight, 1/3)
        return self._weight


