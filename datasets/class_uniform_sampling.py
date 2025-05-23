"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


Uniform sampling of classes.
For all images, for all classes, generate centroids around which to sample.

All images are divided into tiles.
For each tile, a class can be present or not. If it is
present, calculate the centroid of the class and record it.

We would like to thank Peter Kontschieder for the inspiration of this idea.
"""

import sys
import os
import json
import numpy as np

import torch

from collections import defaultdict
from scipy.ndimage.measurements import center_of_mass
from PIL import Image
from tqdm import tqdm

pbar = None


def calc_tile_locations(tile_size, image_size):
    """
    Divide an image into tiles to help us cover classes that are spread out.
    tile_size: size of tile to distribute
    image_size: original image size
    return: locations of the tiles
    """
    image_size_y, image_size_x = image_size
    locations = []
    for y in range(image_size_y // tile_size):
        for x in range(image_size_x // tile_size):
            x_offs = x * tile_size
            y_offs = y * tile_size
            locations.append((x_offs, y_offs))
    return locations


def class_centroids_image(item, tile_size, num_classes, id2trainid):
    """
    For one image, calculate centroids for all classes present in image.
    item: image, image_name
    tile_size:
    num_classes:
    id2trainid: mapping from original id to training ids
    return: Centroids are calculated for each tile.
    """
    image_fn, label_fn = item
    centroids = defaultdict(list)
    mask = np.array(Image.open(label_fn))
    image_size = mask.shape
    tile_locations = calc_tile_locations(tile_size, image_size)

    drop_mask = np.zeros((1024,2048))
    drop_mask[15:840, 14:2030] = 1.0

    mask_copy = mask.copy()
    if id2trainid:
        for k, v in id2trainid.items():
            binary_mask = (mask_copy == k)
            mask[binary_mask] = v

    for x_offs, y_offs in tile_locations:
        patch = mask[y_offs:y_offs + tile_size, x_offs:x_offs + tile_size]
        for class_id in range(num_classes):
            if class_id in patch:
                patch_class = (patch == class_id).astype(int)
                centroid_y, centroid_x = center_of_mass(patch_class)
                centroid_y = int(centroid_y) + y_offs
                centroid_x = int(centroid_x) + x_offs
                centroid = (centroid_x, centroid_y)
                centroids[class_id].append((image_fn, label_fn, centroid,
                                            class_id))
    pbar.update(1)
    return centroids


def pooled_class_centroids_all(items, num_classes, id2trainid, tile_size=1024):
    """
    Calculate class centroids for all classes for all images for all tiles.
    items: list of (image_fn, label_fn)
    tile size: size of tile
    returns: dict that contains a list of centroids for each class
    """
    from multiprocessing.dummy import Pool
    from functools import partial
    pool = Pool(4)
    global pbar
    pbar = tqdm(total=len(items), desc='pooled centroid extraction', file=sys.stdout)
    class_centroids_item = partial(class_centroids_image,
                                   num_classes=num_classes,
                                   id2trainid=id2trainid,
                                   tile_size=tile_size)

    centroids = defaultdict(list)
    new_centroids = pool.map(class_centroids_item, items)
    pool.close()
    pool.join()
    pbar.close()

    # combine each image's items into a single global dict
    for image_items in new_centroids:
        for class_id in image_items:
            centroids[class_id].extend(image_items[class_id])
    return centroids


def unpooled_class_centroids_all(items, num_classes, id2trainid,
                                 tile_size=1024):
    """
    Calculate class centroids for all classes for all images for all tiles.
    items: list of (image_fn, label_fn)
    tile size: size of tile
    returns: dict that contains a list of centroids for each class
    """
    centroids = defaultdict(list)
    global pbar
    pbar = tqdm(total=len(items), desc='centroid extraction', file=sys.stdout)
    for image, label in items:
        new_centroids = class_centroids_image(item=(image, label),
                                              tile_size=tile_size,
                                              num_classes=num_classes,
                                              id2trainid=id2trainid)
        for class_id in new_centroids:
            centroids[class_id].extend(new_centroids[class_id])

    return centroids


def class_centroids_all(items, num_classes, id2trainid, tile_size=1024):
    """
    intermediate function to call pooled_class_centroid
    """
    pooled_centroids = pooled_class_centroids_all(items, num_classes,
                                                  id2trainid, tile_size)
    return pooled_centroids


def random_sampling(alist, num, RNG_seed):
    """
    Randomly sample num items from the list
    alist: list of centroids to sample from
    num: can be larger than the list and if so, then wrap around
    return: class uniform samples from the list
    """
    torch.manual_seed(RNG_seed)
    sampling = []
    len_list = len(alist)
    assert len_list, 'len_list is zero!'
    indices=torch.randperm(len_list).tolist()

    for i in range(num):
        item = alist[indices[i % len_list]]
        sampling.append(item)
    return sampling


def build_centroids(imgs, num_classes, id2trainid=None,json_fn="cityscapes/cityscapes_centroids.json"):
    """
    The first step of uniform sampling is to decide sampling centers.
    The idea is to divide each image into tiles and within each tile,
    we compute a centroid for each class to indicate roughly where to
    sample a crop during training.

    This function computes these centroids and returns a list of them.
    """
    if os.path.isfile(json_fn):
        print('Loading centroid file {}'.format(json_fn))
        with open(json_fn, 'r') as json_data:
            centroids = json.load(json_data)
        centroids = {int(idx): centroids[idx] for idx in centroids}
        print('Found {} centroids'.format(len(centroids)))
    else:
        print('Didn\'t find {}, so building it'.format(json_fn))

        # centroids is a dict (indexed by class) of lists of centroids
        centroids = class_centroids_all(
            imgs,
            num_classes,
            id2trainid=id2trainid)
        with open(json_fn, 'w') as outfile:
            json.dump(centroids, outfile, indent=4)

    return centroids


def build_epoch(imgs, centroids, num_classes, class_uniform_pct, RNG_seed, nb_shots):
    imgs_uniform = []
    for class_id in range(num_classes):
        centroid_len = len(centroids[class_id])
        if centroid_len == 0:
            pass
        else:
            class_centroids = random_sampling(centroids[class_id],
                                              nb_shots, RNG_seed)
            imgs_uniform.extend(class_centroids)
    return imgs_uniform