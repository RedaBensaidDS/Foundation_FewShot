from losses import BootstrappedCE
from lr_schedulers import poly_lr_scheduler,cosine_lr_scheduler,step_lr_scheduler,exp_lr_scheduler
from data import get_cityscapes, build_val_transform,Cityscapes,get_coco, get_ppdls
import torch

def get_lr_function(config,total_iterations):
    # get the learning rate multiplier function for LambdaLR
    name=config["lr_scheduler"]
    warmup_iters=config["warmup_iters"]
    warmup_factor=config["warmup_factor"]
    if "poly"==name:
        p=config["poly_power"]
        return lambda x : poly_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor,p)
    elif "cosine"==name:
        return lambda x : cosine_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor)
    elif "step"==name:
        return lambda x : step_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor)
    elif "exp"==name:
        beta=config["exp_beta"]
        return lambda x : exp_lr_scheduler(x,total_iterations,warmup_iters,warmup_factor,beta)
    else:
        raise NotImplementedError()

def get_loss_fun(config):
    train_crop_size=config["train_crop_size"]
    ignore_value=config["ignore_value"]
    if isinstance(train_crop_size,int):
        crop_h,crop_w=train_crop_size,train_crop_size
    else:
        crop_h,crop_w=train_crop_size
    loss_type="cross_entropy"
    if "loss_type" in config:
        loss_type=config["loss_type"]
    if loss_type=="cross_entropy":
        loss_fun=torch.nn.CrossEntropyLoss(ignore_index=ignore_value)
    elif loss_type=="bootstrapped":
        # 8*768*768/16
        minK=int(config["batch_size"]*crop_h*crop_w/16)
        print(f"bootstrapped minK: {minK}")
        loss_fun=BootstrappedCE(minK,0.3,ignore_index=ignore_value)
    else:
        raise NotImplementedError()
    return loss_fun

def get_optimizer(model,config):
    if not config["bn_weight_decay"]:
        p_bn = [p for n, p in model.named_parameters() if "bn" in n]
        p_non_bn = [p for n, p in model.named_parameters() if "bn" not in n]
        optim_params = [
            {"params": p_bn, "weight_decay": 0},
            {"params": p_non_bn, "weight_decay": config["weight_decay"]},
        ]
    else:
        optim_params = model.parameters()
    return torch.optim.SGD(
        optim_params,
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

def get_val_dataset(config):
    val_input_size=config["val_input_size"]
    val_label_size=config["val_label_size"]

    root=config["dataset_dir"]
    name=config["dataset_name"]
    val_split=config["val_split"]
    if name=="cityscapes":
        val_transform=build_val_transform(val_input_size,val_label_size)
        val = Cityscapes(root, split=val_split, target_type="semantic",
                         transforms=val_transform, class_uniform_pct=0)
    else:
        raise NotImplementedError()
    return val

def get_dataset_loaders(config):
    name=config["dataset_name"]
    if name=="cityscapes":
        train_loader, val_loader,train_set=get_cityscapes(
            config["dataset_dir"],
            config["batch_size"],
            config["train_min_size"],
            config["train_max_size"],
            config["train_crop_size"],
            config["val_input_size"],
            config["val_label_size"],
            config["aug_mode"],
            config["class_uniform_pct"],
            config["train_split"],
            config["val_split"],
            config["num_workers"],
            config["ignore_value"],
            config["RNG_seed"],
            config["number_of_shots"]
        )
    elif name == "ppdls" : 
        train_loader, val_loader,train_set=get_ppdls(
            config["dataset_dir"],
            config["batch_size"],
            config["train_min_size"],
            config["train_max_size"],
            config["train_crop_size"],
            config["val_input_size"],
            config["val_label_size"],
            config["aug_mode"],
            config["num_workers"],
            config["ignore_value"],
            config["RNG_seed"],
            config["split_path"],
            config["number_of_shots"]
        )
    elif name=="coco":
        train_loader, val_loader,train_set=get_coco(
            config["dataset_dir"],
            config["batch_size"],
            config["train_min_size"],
            config["train_max_size"],
            config["train_crop_size"],
            config["val_input_size"],
            config["val_label_size"],
            config["aug_mode"],
            config["num_workers"],
            config["ignore_value"],
            config["RNG_seed"],
            config["number_of_shots"]
        )
    else:
        raise NotImplementedError()
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader, val_loader,train_set

