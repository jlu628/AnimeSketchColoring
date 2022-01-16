import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.input_dir = os.path.join(self.root_dir, "sketch")
        self.target_dir = os.path.join(self.root_dir, "anime")
        self.input_files = os.listdir(self.input_dir)
        self.target_files = os.listdir(self.target_dir)


    def __len__(self):
        return len(self.input_files)


    def __getitem__(self, index):
        input_path = os.path.join(self.input_dir, str(index)+".png")
        target_path = os.path.join(self.target_dir, str(index)+".png")
        input_image = np.array(Image.open(input_path))
        target_image = np.array(Image.open(target_path))
    
        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image