import torch 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        return codebook_mapping, codebook_indices.reshape(codebook_mapping.shape[0], -1)
    
    def gamma_func(self, mode="linear"): # r mean step of iteration, output mean how much ratio of mask should be regenerated
        gamma_funcs = {
            "linear": lambda r: 1 - r,
            "cosine": lambda r: np.cos(r * np.pi / 2),
            "square": lambda r: 1 - r ** 2,
            "sqrt": lambda r: 1 - np.sqrt(r),
            "sine_linear": lambda r: (1 - r) * (1 - np.sin(2 * np.pi * r)  *  0.75),
            "constant": lambda r: np.zeros_like(r)
        }
        if mode not in gamma_funcs:
            raise NotImplementedError
        return gamma_funcs[mode]

    def forward(self, x):
        _, z_indices = self.encode_to_z(x)
        mask_token = torch.ones(z_indices.shape, device=z_indices.device).long() * self.mask_token_id
        mask = torch.bernoulli(torch.rand(z_indices.shape, device=z_indices.device) * 0.75 + 0.25).bool()
        new_indices = mask * mask_token + (~mask) * z_indices
        logits = self.transformer(new_indices)
        return logits, z_indices
    
    @torch.no_grad()
    def inpainting(self, z_indices_predict, mask_bc, mask_num, step_ratio, mask_func):
        z_indices_with_mask = mask_bc * self.mask_token_id + (~mask_bc) * z_indices_predict
        logits = self.transformer(z_indices_with_mask)
        logits = logits[0]
        # print('logits', logits.shape)
        #FIND MAX probability for each token value
        # 使用 softmax 將 logits 轉換為概率分佈
        probs = F.softmax(logits, dim=-1)
        
        # 將 mask_token_id 的概率設為 0，確保不會被採樣到
        probs[:, self.mask_token_id] = 0
        
        # 重新歸一化概率分佈
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # 從修改後的概率分佈中採樣
        z_indices_predict = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # 計算每個 token 的預測機率
        z_indices_predict_prob = probs.gather(1, z_indices_predict.unsqueeze(-1)).squeeze(-1)
        
        # 將不在 mask 中的 token 機率設為無限大
        # 這確保了非 mask 區域的 token 不會被更改
        z_indices_predict_prob = torch.where(
            mask_bc,
            z_indices_predict_prob,
            torch.tensor(float('inf'), device=z_indices_predict_prob.device)
        )


        mask_ratio = self.gamma_func(mask_func)(step_ratio)
        mask_len = torch.floor(mask_num * mask_ratio).long()
        #predicted probabilities add temperature annealing gumbel noise as confidence
        g = torch.distributions.gumbel.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices_predict_prob.device)

        temperature = self.choice_temperature * (1 - step_ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        sorted_confidence = torch.sort(confidence, dim=-1)[0]
        cut_off = sorted_confidence[:, mask_len].unsqueeze(-1)
        new_mask = (confidence < cut_off)
        return z_indices_predict, new_mask
    
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
