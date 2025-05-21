from pathlib import Path
import typing
import torch
from natsort import natsorted
from PIL import Image
import numpy as np
import random
import pickle

def _load_plant_dir(plant_dir):
    rgb_images = plant_dir.glob("plant*_rgb.png")
    rgb_images = natsorted(rgb_images)
    seg_images = [plant_dir / (rgb_path.name.split('_')[0] + "_fg.png")
                  for rgb_path in rgb_images]
    return rgb_images, seg_images

class PPDLS(torch.utils.data.Dataset):
    """Dataset for the leaf segmentation
    from the LEAF COUNTING CHALLENGE.
    """
    def __init__(self, base_path: typing.Union[str, Path], split_path = "", split = "", transforms = None, RNG_seed = 0, nb_shots = 1):
        """
        Loads the dataset.
        Args:
            base_path: The dataset base path
        """
        super().__init__()
        self.transforms = transforms
        base_path = Path(base_path)
        if split_path != "" :
            with open(split_path, 'rb') as fp:
                split_idx = pickle.load(fp)
        random.seed(RNG_seed)
        support_indexes = random.sample(split_idx[0], 2 * nb_shots)

        a1_rgb, a1_mask = _load_plant_dir(base_path / "A1")
        a2_rgb, a2_mask = _load_plant_dir(base_path / "A2")
        a4_rgb, a4_mask = _load_plant_dir(base_path / "A4")
        self.rgb_images = a1_rgb + a2_rgb + a4_rgb
        self.seg_images = a1_mask + a2_mask + a4_mask
        if split == "train" :

            self.rgb_images = [self.rgb_images[i] for i in support_indexes]
            self.seg_images = [self.seg_images[i] for i in support_indexes]
            print(self.rgb_images)
            print(len(self.rgb_images))
            print(len(self.seg_images))
        if split == "validation" :
            self.rgb_images = [self.rgb_images[i] for i in split_idx[1]]
            self.seg_images = [self.seg_images[i] for i in split_idx[1]]
    def __getitem__(self, idx: int) :
        """Loads a dataset image and mask.
        Args:
            idx: Index
        """
        image = Image.open(self.rgb_images[idx]).convert("RGB")

        raw_mask_image = np.array(Image.open(self.seg_images[idx]))
        mask = raw_mask_image > 0 
        mask_image = np.zeros_like(raw_mask_image, dtype=np.int8)
        mask_image[mask] = 1
        target = Image.fromarray(mask_image, mode = "L")

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return  image, target
    def __len__(self):
        """
        Dataset size.
        """
        return len(self.rgb_images)