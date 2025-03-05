import math
import torch
# import deepspeed
# import random

# import numpy as np
import models.decoders
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as  nn
from models.model_config import DATASETS_NAME_TO_INTERPRETER
from transformers import LlavaOnevisionForConditionalGeneration
from timm.models import create_model
from timm.models.registry import register_model


# def get_rank():
#     try: return deepspeed.dist.get_rank()
#     except: return 0

class GumbelVQ(nn.Module):
    def __init__(self,
            in_channel, 
            codebook_size=64, 
            embedding_dim=128,
            max_steps = 50000,
            initial_t=2.0, 
            final_t=0.1,
            start_iters = 0):
        super().__init__()
        self.step = start_iters
        self.max_steps = max_steps
        self.initial_t = initial_t
        self.final_t = final_t

        self.temperature = self.cosine_decay()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(codebook_size, embedding_dim)
        self.pre_proj = nn.Linear(in_channel, codebook_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Embedding
        init.normal_(self.embed.weight, mean=0, std=0.2)
        # Initialize nn.Linear (pre_proj)
        init.xavier_uniform_(self.pre_proj.weight)  # Xavier initialization for the weights
        if self.pre_proj.bias is not None:
            init.zeros_(self.pre_proj.bias)  # Bias to 0
            
    def cosine_decay(self):
        self.step += 1
        if self.step >= self.max_steps: return self.final_t
        return self.final_t + 0.5 * (self.initial_t - self.final_t) * (1 + math.cos(math.pi * self.step / self.max_steps))


    def linear_decay(self):
        self.step += 1
        if self.step >= self.max_steps: return self.final_t
        return self.final_t + (self.initial_t - self.final_t) * (self.max_steps - self.step) / self.max_steps

    def forward(self, logits, temperature = None, hard_forward = False):
        
        if hard_forward: return self.greedy_forward(logits)
        temperature = temperature if temperature is not None else self.temperature
        logits = self.pre_proj(logits)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=False)

        quantized = torch.einsum('b n, n d -> b d', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        entropy_loss = 5e-4 * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()

        self.temperature = self.linear_decay() 
        return quantized, torch.max(soft_one_hot, dim=-1), entropy_loss


    def greedy_forward(self, logits):
        logits = self.pre_proj(logits)
        indices = torch.argmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=logits.size(1)).to(self.embed.weight.device).to(self.embed.weight.dtype)

        quantized = torch.einsum('b n, n d -> b d', one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        entropy_loss = 5e-4 * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()

        return quantized.detach(), torch.max(one_hot, dim=-1), entropy_loss


class UniAct(nn.Module):
    def __init__(self, 
                ua_extractor_name,
                vision_backbone = "resnet18.a1_in1k",
                codebook_size = 256,
                universal_action_dim = 128,
                max_steps = 50000,
                start_iters = 0,
                initial_t = 2.0,
                final_t = 0.1,
                freeze_vision_backbone=False,
                freeze_ua_extractor=False,
                freeze_codebook=False,
                # NOTE: Enable hard-select only when perform fast adapation
                # For in-domain test(like Libero), hard-selection will lead to slight performance degredation 
                hard_forward=False
                ):
        super().__init__()


        # initialize base model
        self.ua_extractor = LlavaOnevisionForConditionalGeneration.from_pretrained(
            ua_extractor_name,
            torch_dtype='auto', 
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"
            )

        # initialize universal action codebook
        self.universal_actions_codebook = GumbelVQ(
                            in_channel=self.ua_extractor.language_model.config.hidden_size,
                            codebook_size=codebook_size,
                            embedding_dim=universal_action_dim,
                            max_steps=max_steps,
                            start_iters = start_iters,
                            initial_t = initial_t,
                            final_t = final_t)


        self.vision_backbone = create_model(
            vision_backbone,
            pretrained=True,
            num_classes=0,
        )
        
        self.freeze_ua_extractor = freeze_ua_extractor
        self.freeze_vision_backbone = freeze_vision_backbone
        self.freeze_codebook = freeze_codebook
        self.hard_forward = hard_forward
        if freeze_vision_backbone: 
            self.vision_backbone.requires_grad_(False)
            
        if self.freeze_ua_extractor:
            self.ua_extractor.requires_grad_(False)
            
        if self.freeze_codebook:
            self.universal_actions_codebook.requires_grad_(False)

        # initialize embodiment-specific low-level interpreters
        self.interpreter = nn.ModuleDict()
        for domain_name, interpreter in DATASETS_NAME_TO_INTERPRETER.items(): 
                self.interpreter[domain_name] = create_model(interpreter)
        self.loss = nn.HuberLoss(delta=0.1, reduction='none')
        
        print("=============model loaded===========")
    
    def get_vision_embedding(self, 
                           vision_input: torch.Tensor): # B V 3 H W
        B, V, C, H, W = vision_input.shape
        vision_embedding = self.vision_backbone.forward_features(vision_input.view(B*V, C, H, W)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).view(B, V, N, num_features)
        return vision_embedding


    def forward(self, 
                domain_name,
                inputs,
                images,
                action,
                action_mask,
                
                # Optioinal
                log_file: str = None,
                proprios = None,
                ):
        
        if self.freeze_ua_extractor:
            self.ua_extractor.eval()
            with torch.no_grad():
                output_hidden_states = self.ua_extractor(**inputs, output_hidden_states=True, return_dict = True).hidden_states[-1]
                last_indices = (inputs.attention_mask.cumsum(dim=1) * inputs.attention_mask).argmax(dim=1).to(output_hidden_states.device)
                universal_action = output_hidden_states[torch.arange(len(output_hidden_states), device=output_hidden_states.device), last_indices]
        else:
                output_hidden_states = self.ua_extractor(**inputs, output_hidden_states=True, return_dict = True).hidden_states[-1]
                last_indices = (inputs.attention_mask.cumsum(dim=1) * inputs.attention_mask).argmax(dim=1).to(output_hidden_states.device)
                universal_action = output_hidden_states[torch.arange(len(output_hidden_states), device=output_hidden_states.device), last_indices]
        
        universal_action, (max_confidence, max_index), entropy_loss = \
            self.universal_actions_codebook(universal_action, hard_forward = self.hard_forward)


        # NOTE: for debug, avoiding codebook collapse
        # if get_rank() == 0: 
        #     if random.random() < 0.1:
        #         with open(log_file, "a+") as f:
        #             f.write(f"{max_confidence},{max_index},domain_name:{domain_name} \n")

        interpreter = self.interpreter[str(domain_name)]
        pred = interpreter(vision_embedding=self.get_vision_embedding(images), 
                           universal_action=universal_action, 
                           proprios = proprios)
        
        action_loss =  (self.loss(pred, action) * action_mask).sum() / action_mask.sum()
        
        return action_loss, \
                {f"{domain_name}_loss": action_loss.item(), 
                "entropy_loss": entropy_loss.item(),
                'gumbel_temp': self.universal_actions_codebook.temperature}



    @torch.no_grad()
    def infer(self, 
                domain_name,
                inputs,
                images,
                
                # Optioinal
                proprios = None,
                ):
        
        output_hidden_states = self.ua_extractor(**inputs, output_hidden_states=True, return_dict = True).hidden_states[-1]
        last_indices = (inputs.attention_mask.cumsum(dim=1) * inputs.attention_mask).argmax(dim=1).to(output_hidden_states.device)
        universal_action = output_hidden_states[torch.arange(len(output_hidden_states), device=output_hidden_states.device), last_indices]

        
        universal_action, (max_confidence, max_index), entropy_loss = \
            self.universal_actions_codebook(universal_action, hard_forward = self.hard_forward)


        interpreter = self.interpreter[str(domain_name)]
        pred = interpreter(vision_embedding=self.get_vision_embedding(images), 
                           universal_action=universal_action, 
                           proprios = proprios)
        
        return pred



@register_model
def UniAct_05B_CodeBook_256_V1(**kwargs):
    return UniAct(ua_extractor_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
                vision_backbone = "resnet18.a1_in1k",
                codebook_size = 256,
                universal_action_dim = 128,
                # NOTE: this is a hyper-parameters for inference and 1.4 is aligned with training
                initial_t = 1.4, 
                final_t = 1.4)


@register_model
def UniAct_05B_CodeBook_256_V1_For_Fast_Adaptation(**kwargs):
    return UniAct(ua_extractor_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
                vision_backbone = "resnet18.a1_in1k",
                codebook_size = 256,
                universal_action_dim = 128,
                # NOTE: freeze ua extractor and codebook 
                freeze_vision_backbone = False,
                freeze_ua_extractor = True,
                freeze_codebook = True,
                hard_forward = True)


@register_model
def UniAct_05B_CodeBook_256_V1_Fulltune(max_steps = 1000000, 
                            start_iters = 0,  
                            initial_t = 2.0, 
                            final_t = 0.5, 
                            **kwargs):
    return UniAct(ua_extractor_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
                vision_backbone = "resnet18.a1_in1k",
                codebook_size = 256,
                universal_action_dim = 128,
                max_steps = max_steps,
                start_iters = start_iters,
                initial_t = initial_t,
                final_t = final_t,
                freeze_vision_backbone = False,
                freeze_ua_extractor = False,
                freeze_codebook = False)
