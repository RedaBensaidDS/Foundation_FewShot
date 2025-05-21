
import torch
import torch.nn as nn 
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import SegformerForSemanticSegmentation
from svf import *

class SEGFORMER_linear(nn.Module):
    def __init__(self, method, num_classes, model_path):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", cache_dir = model_path)
        self.encoder = self.model.segformer.encoder
        self.method = method
        if self.method != "finetune":
            self.encoder.eval()
        if method == "svf" : 
            self.encoder = resolver(self.encoder)
        if method == "lora" : 
            config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "key", "value"],
                lora_dropout=0.1,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, config)
        if method == "multilayer" : 
            self.in_channels = 512 
        else : 
            self.in_channels = 256 
        self.decoder = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        self.bn = nn.SyncBatchNorm(self.in_channels)

    def forward(self, x): 
        if self.method in ["linear", "multilayer"] : 
            with torch.no_grad():
                x = F.interpolate(x, size=[1024,1024], mode='bilinear', align_corners=False)
                if self.method == "multilayer" : 
                    x = self.encoder(x, output_hidden_states = True, return_dict = True)["hidden_states"]
                    outputs = []
                    for i, feature in enumerate(x) :
                        outputs.append(F.interpolate(feature, size=[256,256], mode='bilinear', align_corners=False))
                    x = torch.cat(outputs,dim=1)
                else : 
                    x = self.encoder(x).last_hidden_state
        else : 
            x = F.interpolate(x, size=[1024,1024], mode='bilinear', align_corners=False)
            x = self.encoder(x).last_hidden_state

        x = self.bn(x)
        return self.decoder(x)