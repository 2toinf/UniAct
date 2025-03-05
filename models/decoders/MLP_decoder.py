from timm.models.registry import register_model
from timm.models.layers import Mlp
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MLP_Decoder(nn.Module):
    def __init__(self,
                universal_action_dim = 128,
                hidden_dim = 512,
                action_dim = 7,
                action_chunking_length = 4):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunking_length = action_chunking_length
        self.head = Mlp(in_features=hidden_dim + universal_action_dim, 
                        hidden_features=action_dim * action_chunking_length * 4, 
                        out_features=action_dim * action_chunking_length)


    def forward(self, 
                vision_embedding: torch.Tensor,  # B V N C
                universal_action: torch.Tensor, # B, ua_dim
                **kwargs): # B, prio_dim
        B = vision_embedding.shape[0]
        inputs = torch.mean(torch.flatten(vision_embedding, 1, 2), dim = 1)
        inputs = torch.cat((inputs, universal_action), dim = -1)
        pred = self.head(inputs).view(B, self.action_chunking_length, self.action_dim) # B, action_dim
        return pred


@register_model
def MLP_1RGB_7DoFs_4FutureAction(**kwargs):
    return MLP_Decoder(
        universal_action_dim = 128,
        hidden_dim = 512,
        action_dim = 7,
        action_chunking_length = 4
    )









