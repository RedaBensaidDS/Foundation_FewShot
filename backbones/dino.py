import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from functools import partial
from peft import LoraConfig, get_peft_model
from svf import *

def create_backbone_dino(method, model_repo_path, model_path) : 
    sys.path.insert(0,os.path.join(model_repo_path, "dino"))
    from vision_transformer import vit_base, vit_small
    dino_backbone = vit_small(patch_size = 16)
    dino_backbone.load_state_dict(torch.load(os.path.join(model_path, "dino_vitbase16_pretrain.pth")))
    if method == "multilayer" :  
        n = 4
    else : 
        n = 1
    dino_backbone.forward = partial(
            dino_backbone.get_intermediate_layers,
            n=n,
        )
    return dino_backbone

def create_backbone_dinov2(method, model_repo_path, model_path) : 
    sys.path.insert(0,os.path.join(model_repo_path, "dinov2"))
    from dinov2.models.vision_transformer import vit_base, vit_large, vit_small
    dino_backbone = vit_base(patch_size = 14, img_size = 524, init_values=1.0, block_chunks=0, num_register_tokens=0) 
    dino_backbone.load_state_dict(torch.load(os.path.join(model_path,"dinov2_vitb14_pretrain.pth"))) 
    if method == "multilayer" :
        n = [8,9,10,11]
    else : 
        n = [11]
    dino_backbone.forward = partial(
            dino_backbone.get_intermediate_layers,
            n=n,
            reshape=True,
        )
    return dino_backbone

class DINO_linear(nn.Module):
    def __init__(self, version, method, num_classes, input_size, model_repo_path, model_path):
        super().__init__()
        self.method = method
        self.version = version
        self.input_size = input_size
        if self.version == 2 : 
            self.encoder = create_backbone_dinov2(method, model_repo_path, model_path)
        else : 
            self.encoder = create_backbone_dino(method, model_repo_path, model_path)
        if self.method == "vpt" : 
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.register_tokens.requires_grad = True
        if method == "svf" : 
            self.encoder = resolver(self.encoder)
        if method == "lora" : 
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["qkv"],
                lora_dropout=0.1,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, config)
        if method == "multilayer" : 
            self.in_channels = 768*4
        else : 
            self.in_channels = 768
        self.decoder = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        self.bn = nn.SyncBatchNorm(self.in_channels)

    def forward(self, x): 
        if self.version == 2 : 
            input_dim = int(x.shape[-1]/14)*14
        else : 
            input_dim = x.shape[-1]
        input_dim = int(self.input_size/14)*14
        if self.method in ["linear", "multilayer"] : 
            with torch.no_grad():
                x = F.interpolate(x, size=[input_dim,input_dim], mode='bilinear', align_corners=False)
                x = self.encoder(x)
        else : 
            x = F.interpolate(x, size=[input_dim,input_dim], mode='bilinear', align_corners=False)
            x = self.encoder(x)
    
        x = torch.cat(x,dim=1)
        if self.version == 1 : 
            x = x[:,1:,:]
            x = x.reshape(x.shape[0], int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
        x = self.bn(x)
        return self.decoder(x)
