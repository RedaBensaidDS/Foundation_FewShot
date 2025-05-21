import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
import os
import sys
from peft import LoraConfig, get_peft_model
from svf import *

def create_backbone_ibot(method, model_repo_path, model_path) : 
    sys.path.insert(0,os.path.join(model_repo_path, "ibot"))
    from models.vision_transformer import vit_base
    state_dict = torch.load(os.path.join(model_path, "checkpoint_teacher.pth"))['state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    vit = vit_base(patch_size=16, return_all_tokens=True).cuda()
    vit.load_state_dict(state_dict, strict=False)
    vit.eval()
    if method == "multilayer" : 
        n = 4 
    else : 
        n = 1
    vit.forward = partial(
            vit.get_intermediate_layers,
            n=n,
        )    
    return vit

class IBOT_linear(nn.Module):
    def __init__(self,method, num_classes, model_repo_path, model_path):
        super().__init__()
        self.method = method
        self.encoder = create_backbone_ibot(method, model_repo_path, model_path)
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
        if self.method in ["linear", "multilayer"] : 
            with torch.no_grad():
                x = F.interpolate(x, size=[1024,1024], mode='bilinear', align_corners=False)
                outputs = self.encoder(x)
        else : 
            x = F.interpolate(x, size=[1024,1024], mode='bilinear', align_corners=False)
            outputs = self.encoder(x)

        outputs = [out[:, 1:] for out in outputs]
        outputs = [
            out.reshape(x.shape[0],  int(np.sqrt(out.shape[1])),  int(np.sqrt(out.shape[1])), -1).permute(0, 3, 1, 2).contiguous()
            for out in outputs
        ]
        outputs = torch.cat(outputs,dim=1)
        outputs = self.bn(outputs)
        return self.decoder(outputs)

