import datetime
import time
import torch
import yaml
import torch.cuda.amp as amp
import os
import copy
import random
import numpy as np
import torch.nn.functional as F
from train_utils import get_lr_function, get_loss_fun,get_optimizer,get_dataset_loaders,get_val_dataset
from precise_bn import compute_precise_bn_stats
import warnings
warnings.filterwarnings("ignore")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class ConfusionMatrix(object):
    def __init__(self, num_classes, exclude_classes):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.exclude_classes=exclude_classes

    def update(self, a, b):
        a=a.cpu()
        b=b.cpu()
        n = self.num_classes
        k = (a >= 0) & (a < n)
        inds = n * a + b
        inds=inds[k]
        self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        acc_global=acc_global.item() * 100
        acc=(acc * 100).tolist()
        iu=(iu * 100).tolist()
        return acc_global, acc, iu
    def __str__(self):
        acc_global, acc, iu = self.compute()
        acc_global=round(acc_global,2)
        IOU=[round(i,2) for i in iu]
        mIOU=sum(iu)/len(iu)
        mIOU=round(mIOU,2)
        reduced_iu=[iu[i] for i in range(self.num_classes) if i not in self.exclude_classes]
        mIOU_reduced=sum(reduced_iu)/len(reduced_iu)
        mIOU_reduced=round(mIOU_reduced,2)
        return f"IOU: {IOU}\nmIOU: {mIOU}, mIOU_reduced: {mIOU_reduced}, accuracy: {acc_global}"

def evaluate(model, data_loader, device, confmat,mixed_precision,print_every,max_eval):
    model.eval()
    assert(isinstance(confmat,ConfusionMatrix))
    with torch.no_grad():
        for i,(image, target) in enumerate(data_loader):
            if (i+1)%print_every==0:
                print(i+1)
            image, target = image.to(device), target.to(device)
            with amp.autocast(enabled=mixed_precision):
                output = model(image)
            output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
            confmat.update(target.flatten(), output.argmax(1).flatten())
            if i+1==max_eval:
                break
    return confmat

def train_one_epoch(model, loss_fun, optimizer, loader, lr_scheduler, print_every, mixed_precision, scaler):
    model.train()
    losses=0
    for t, x in enumerate(loader):
        image, target=x
        image, target = image.to("cuda"), target.to("cuda")
        with amp.autocast(enabled=mixed_precision):
            output = model(image)
            output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
            loss = loss_fun(output, torch.tensor(target,dtype = torch.int64))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        losses+=loss.item()
        if (t+1) % print_every==0:
            print(t+1,loss.item())
    num_iter=len(loader)
    print(losses/num_iter)
    return losses/num_iter

def get_epochs_to_eval(config):
    epochs=config["epochs"]
    eval_every_k_epochs=config["eval_every_k_epochs"]
    eval_best_on_epochs=[i*eval_every_k_epochs-1 for i in range(1,epochs//eval_every_k_epochs+1)]
    if epochs-1 not in eval_best_on_epochs:
        eval_best_on_epochs.append(epochs-1)
    if 0 not in eval_best_on_epochs:
        eval_best_on_epochs.append(0)
    if "eval_last_k_epochs" in config:
        for i in range(max(epochs-config["eval_last_k_epochs"],0),epochs):
            if i not in eval_best_on_epochs:
                eval_best_on_epochs.append(i)
    eval_best_on_epochs=sorted(eval_best_on_epochs)
    return eval_best_on_epochs
def setup_env(config):
    torch.backends.cudnn.benchmark=True
    seed=0
    if "RNG_seed" in config:
        seed=config["RNG_seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) 
def train_multiple(configs, model, method, dataset, number_of_shots):
    global_accuracies=[]
    mIOUs=[]
    new_configs=[]
    for config in configs:
        new_configs.append(config)
    for i, config in enumerate(new_configs):
        config["model"] = model
        config["method"] = method
        config["dataset"] = dataset
        config["number_of_shots"] = number_of_shots
        learning_rate = config["lr"]

        run = config["run"]
        print(f"START RUN {run}")
        best_mIU,best_global_accuracy=train_one(config, model, method, dataset, number_of_shots)
        mIOUs.append(best_mIU)
        global_accuracies.append(best_global_accuracy)
    return mIOUs,global_accuracies
def train_one(config, model_str, method, dataset, number_of_shots):
    setup_env(config)
    save_dir = config["save_dir"]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs=config["epochs"]
    num_classes=config["num_classes"] 
    exclude_classes=config["exclude_classes"]
    mixed_precision=config["mixed_precision"]
    run=config["run"]
    max_eval=config["max_eval"]
    eval_print_every=config["eval_print_every"]
    train_print_every=config["train_print_every"]
    bn_precise_stats=config["bn_precise_stats"]
    bn_precise_num_samples=config["bn_precise_num_samples"]
    input_size = config["input_size"]
    print("input size is ", input_size)

    if model_str == "SEGFORMER" : 
        from backbones.segformer import SEGFORMER_linear
        model = SEGFORMER_linear(method, num_classes, config["model_path"])
    if model_str == "IBOT" :
        from backbones.ibot import IBOT_linear
        model = IBOT_linear(method, num_classes, config["model_repo_path"], config["model_path"])
    if model_str == "SAM" : 
        from backbones.sam import SAM_linear
        model = SAM_linear(method, num_classes, config["model_repo_path"], config["model_path"])
    if model_str == "CLIP" : 
        from backbones.clip import CLIP_linear
        model = CLIP_linear(method, num_classes, config["model_path"])
    if model_str == "DINO" :
        from backbones.dino import DINO_linear
        version = 2 #dinov2
        model = DINO_linear(version, method, num_classes, input_size, config["model_repo_path"], config["model_path"])
    if model_str == "ResNet" : 
        from backbones.resnet import ResNet_linear
        model = ResNet_linear(method, num_classes, config["model_path"])
    if model_str == "MAE" :
        from backbones.mae import MAE_linear 
        model = MAE_linear(method, num_classes, config["model_repo_path"], config["model_path"])

    print_trainable_parameters(model)
    print_trainable_parameters(model.encoder)

    if method in ["svf", "finetune", "lora", "vpt"] :
        model_tmp = torch.load(f"{save_dir}/{model_str}_linear_{dataset}_{number_of_shots}shot_run{run}.pth")
        model.decoder = model_tmp.decoder
        model.bn = model_tmp.bn
        del model_tmp
    model.to(device=device)

    train_loader, val_loader,train_set=get_dataset_loaders(config)
    total_iterations=len(train_loader) * epochs
    optimizer = get_optimizer(model,config)
    scaler = amp.GradScaler(enabled=mixed_precision)
    loss_fun=get_loss_fun(config)
    lr_function=get_lr_function(config,total_iterations)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters = total_iterations, power = config["poly_power"])
    epoch_start=0
    best_mIU=0
    eval_on_epochs=get_epochs_to_eval(config)
    print("eval on epochs: ",eval_on_epochs)
    start_time = time.time()
    best_global_accuracy=0
    for epoch in range(epoch_start,epochs):
        print(f"epoch: {epoch}")
        if hasattr(train_set, 'build_epoch'):
            print("build epoch")
            train_set.build_epoch()
        average_loss=train_one_epoch(model, loss_fun, optimizer, train_loader, lr_scheduler, print_every=train_print_every, mixed_precision=mixed_precision, scaler=scaler)
        if epoch in eval_on_epochs:
            if bn_precise_stats:
                print("calculating precise bn stats")
                compute_precise_bn_stats(model,train_loader,bn_precise_num_samples)
            confmat=ConfusionMatrix(num_classes,exclude_classes)
            confmat = evaluate(model, val_loader, device,confmat,
                               mixed_precision, eval_print_every,max_eval)
            print(confmat)
            acc_global, acc, iu = confmat.compute()
            acc_global=round(acc_global,2)
            IOU=[round(i,2) for i in iu]
            mIOU=sum(iu)/len(iu)
            mIOU=round(mIOU,2)
            reduced_iu=[iu[i] for i in range(confmat.num_classes) if i not in confmat.exclude_classes]
            mIOU_reduced=sum(reduced_iu)/len(reduced_iu)
            mIOU_reduced=round(mIOU_reduced,2)
            if acc_global>best_global_accuracy:
                best_global_accuracy=acc_global
            if mIOU > best_mIU:
                best_mIU=mIOU
    if method == "linear" : 
        torch.save(model, f"{save_dir}/{model_str}_{method}_{dataset}_{number_of_shots}shot_run{run}.pth")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Best mIOU: {best_mIU}\n")
    print(f"Best global accuracy: {best_global_accuracy}\n")
    print(f"Training time {total_time_str}\n")
    print(f"Training time {total_time_str}")
    return best_mIU,best_global_accuracy

def train_3runs(model, method, dataset, number_of_shots, learning_rate, input_size):
    print(f'dataset is {dataset}')
    if dataset == "cityscapes" : 
        config_filename= "configs/cityscapes.yaml"
    if dataset == "coco" : 
        config_filename="configs/coco.yaml"
    if dataset == "ppdls" : 
        config_filename = "configs/ppdls.yaml"
    with open(config_filename) as file:
        config=yaml.full_load(file)
    config["lr"] = learning_rate
    config["input_size"] = input_size

    configs=[]
    for run in range(1,4):
        new_config = copy.deepcopy(config)
        new_config["run"] = run
        new_config["RNG_seed"] = run-1
        configs.append(new_config)
    train_multiple(configs, model, method, dataset, number_of_shots)


if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--models", nargs = '+')
    parser.add_argument("--methods", nargs = '+')
    parser.add_argument("--lr", nargs = '+')
    parser.add_argument("--dataset", type = str)
    parser.add_argument("--nb-shots", type = int)
    parser.add_argument("--input-size", type = int, default = 1024)

    
    args = parser.parse_args()
    #lr_list = [1e-2 , 1e-3 , 1e-4 , 1e-5 , 1e-6]
    lr_list = args.lr
    for method in args.methods : 
        for model in args.models :
            for learning_rate in lr_list :  
                print(f"START TRAINING {model} WITH {method} WITH {learning_rate} AND {args.nb_shots} SHOT")
                train_3runs(model, method, args.dataset, args.nb_shots, float(learning_rate), float(args.input_size))