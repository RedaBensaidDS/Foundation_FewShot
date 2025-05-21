
import sys
import os
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from svf import *



def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    import models_mae
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

class MAE_linear(nn.Module):
    def __init__(self, method, num_classes, model_repo_path, model_path):
        super().__init__()
        sys.path.insert(0,os.path.join(model_repo_path, "mae"))
        chkpt_dir = os.path.join(model_path, "mae_pretrain_vit_base.pth")
        model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
        self.encoder = model_mae
        self.method = method
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
            n = [8,9,10,11]
            self.in_channels = 768*4
            self.encoder.forward = partial(self.get_intermediate_layers , n = n, norm = True)
        else : 
            n = [11]
            self.encoder.forward = partial(self.get_intermediate_layers , n = n, norm = True)
            self.in_channels = 768
        self.decoder = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        self.bn = nn.SyncBatchNorm(self.in_channels)

    def forward(self, x): 
        if self.method in ["linear", "multilayer"] : 
            with torch.no_grad():
                x = F.interpolate(x, size=[224,224], mode='bilinear', align_corners=False)
                x = self.encoder(x)
                x = torch.cat(x,dim=1)
        else : 
            x = F.interpolate(x, size=[224,224], mode='bilinear', align_corners=False)
            x = self.encoder(x)[0]
        x = self.bn(x)
        return self.decoder(x)
    
    def get_intermediate_layers(self, x, n, norm = True) :         
        output = []
        self.encoder.patch_embed.img_size = (224, 224)
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :]
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk_index, blk in enumerate(self.encoder.blocks) : 
            x = blk(x)
            if blk_index in n : 
                output.append(x)
        output = [out[:, 1:] for out in output]
        if norm:
            output = [self.encoder.norm(out) for out in output] 
        B, w, h = x.shape
        output = [
            out.reshape(B, int(np.sqrt(w)), int(np.sqrt(w)), h).permute(0, 3, 1, 2).contiguous()
            for out in output
        ]
        return output