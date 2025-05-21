import transforms as T
from data_utils import *
from datasets.cityscapes import Cityscapes
from datasets.coco import Coco
from datasets.ppdls import PPDLS


def build_val_transform(val_input_size,val_label_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms=[]
    transforms.append(

        T.ValResize(val_input_size,val_label_size)
    )
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)

def build_train_transform3():
    transforms=[]
    transforms.append(
        T.ValResize(1024,1024)
    )
    transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.RandomVerticalFlip(0.5))
    transforms.append(T.ColorJitter(0.6, 0.2,0.1, 0.01,prob=1))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def build_val_transform3():
    transforms=[]
    transforms.append(
        T.ValResize(1024,1024)
    )
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def build_train_transform2(train_min_size, train_max_size, train_crop_size, aug_mode,ignore_value):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    fill = tuple([int(v * 255) for v in mean])
    edge_aware_crop=False
    resize_mode="uniform"
    transforms = []
    transforms.append(
        T.RandomResize(train_min_size, train_max_size, resize_mode)
    )
    if isinstance(train_crop_size,int):
        crop_h,crop_w=train_crop_size,train_crop_size
    else:
        crop_h,crop_w=train_crop_size
    transforms.append(
        T.RandomCrop2(crop_h,crop_w,edge_aware=edge_aware_crop)
    )
    transforms.append(T.RandomHorizontalFlip(0.5))
    if aug_mode == "baseline":
        pass
    elif aug_mode == "randaug":
        transforms.append(T.RandAugment(2, 0.2, "full",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode=="randaug_reduced":
        transforms.append(T.RandAugment(2, 0.2, "reduced",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode== "colour_jitter":
        transforms.append(T.ColorJitter(0.3, 0.3,0.3, 0,prob=1))
    elif aug_mode=="rotate":
        transforms.append(T.RandomRotation((-10,10), mean=fill, ignore_value=ignore_value,prob=1.0,expand=False))
    elif aug_mode=="noise":
        transforms.append(T.AddNoise(15,prob=1.0))
    elif aug_mode=="noise2":
        transforms.append(T.AddNoise2(10,prob=1.0))
    elif aug_mode=="noise3":
        transforms.append(T.AddNoise3(10,prob=1.0))
    elif aug_mode == "custom1":
        transforms.append(T.RandAugment(2, 0.2, "reduced",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
        transforms.append(T.AddNoise(10,prob=0.2))
    elif aug_mode == "custom2":
        transforms.append(T.RandAugment(2, 0.2, "reduced2",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
        transforms.append(T.AddNoise(10,prob=0.1))
    elif aug_mode=="custom3":
        transforms.append(T.ColorJitter(0.3, 0.4,0.5, 0,prob=1))
    else:
        raise NotImplementedError()
    transforms.append(T.RandomPad(crop_h,crop_w,fill,ignore_value,random_pad=True))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)
def build_train_transform(train_min_size, train_max_size, train_crop_size, aug_mode,ignore_value):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    fill = tuple([int(v * 255) for v in mean])
    #ignore_value = 255
    edge_aware_crop=False
    resize_mode="uniform"
    transforms = []
    transforms.append(
        T.RandomResize(train_min_size, train_max_size, resize_mode)
    )
    if isinstance(train_crop_size,int):
        crop_h,crop_w=train_crop_size,train_crop_size
    else:
        crop_h,crop_w=train_crop_size
    transforms.append(
        T.RandomCrop(crop_h,crop_w,fill,ignore_value,random_pad=True,edge_aware=edge_aware_crop)
    )
    transforms.append(T.RandomHorizontalFlip(0.5))
    if aug_mode == "baseline":
        pass
    elif aug_mode == "randaug":
        transforms.append(T.RandAugment(2, 0.2, "full",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode=="randaug_reduced":
        transforms.append(T.RandAugment(2, 0.2, "reduced",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode=="randaug_reduced2":
        transforms.append(T.RandAugment(2, 0.3, "reduced2",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode=="randaug_reduced3":
        transforms.append(T.RandAugment(2, 0.3, "reduced",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode== "colour_jitter":
        transforms.append(T.ColorJitter(0.3, 0.3,0.3, 0,prob=1))
    elif aug_mode=="rotate":
        transforms.append(T.RandomRotation((-10,10), mean=fill, ignore_value=ignore_value,prob=1.0,expand=False))
    elif aug_mode=="noise":
        transforms.append(T.AddNoise(10,prob=1.0))
    elif aug_mode == "custom1":
        transforms.append(T.RandAugment(2, 0.2, "reduced",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
        transforms.append(T.AddNoise(10,prob=0.2))
    elif aug_mode == "custom2":
        transforms.append(T.RandAugment(2, 0.2, "reduced2",prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
        transforms.append(T.AddNoise(10,prob=0.1))
    elif aug_mode=="custom3":
        transforms.append(T.ColorJitter(0.3, 0.4,0.5, 0,prob=1))
    else:
        raise NotImplementedError()
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)

def get_cityscapes(root, batch_size, train_min_size, train_max_size, train_crop_size, val_input_size,val_label_size, aug_mode,class_uniform_pct,train_split,val_split,num_workers,ignore_value, RNG_seed, nb_shots):

    train_transform=build_train_transform2(train_min_size, train_max_size, train_crop_size, aug_mode, ignore_value)
    val_transform=build_val_transform(val_input_size,val_label_size)
    train = Cityscapes(root, split=train_split, target_type="semantic",
                       transforms=train_transform, class_uniform_pct=class_uniform_pct, RNG_seed=RNG_seed, nb_shots = nb_shots)
    val = Cityscapes(root, split=val_split, target_type="semantic",
                     transforms=val_transform, class_uniform_pct=class_uniform_pct, RNG_seed=RNG_seed, nb_shots = nb_shots)
    train_loader = get_dataloader_train(train, batch_size, num_workers)
    val_loader = get_dataloader_val(val, num_workers)
    return train_loader, val_loader,train

def get_coco(root, batch_size, train_min_size, train_max_size, train_crop_size, val_input_size, val_label_size, aug_mode, num_workers, ignore_value, RNG_seed, nb_shots):
    train_transform=build_train_transform(train_min_size, train_max_size, train_crop_size, aug_mode, ignore_value)
    val_transform=build_val_transform(val_input_size,val_label_size)
    train = Coco(root, "train",train_transform, RNG_seed, nb_shots)
    val = Coco(root, "val",val_transform, RNG_seed, nb_shots)
    train_loader = get_dataloader_train(train, batch_size, num_workers)
    val_loader = get_dataloader_val(val, num_workers)
    return train_loader, val_loader,train

def get_ppdls(root, batch_size, train_min_size, train_max_size, train_crop_size, val_input_size, val_label_size, aug_mode, num_workers, ignore_value, RNG_seed, split_path, nb_shots):
    train_transform=build_train_transform(train_min_size, train_max_size, train_crop_size, aug_mode, ignore_value)
    val_transform=build_val_transform(val_input_size,val_label_size)
    train = PPDLS(root, split_path, "train", train_transform, RNG_seed, nb_shots)
    val = PPDLS(root, split_path, "validation", val_transform, RNG_seed, nb_shots)
    train_loader = get_dataloader_train(train, batch_size, num_workers)
    val_loader = get_dataloader_val(val, num_workers)
    return train_loader, val_loader,train
