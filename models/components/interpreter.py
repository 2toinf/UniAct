from timm.models.registry import register_model
from timm.models.layers import Mlp
import torch
import torch.nn as nn

import torch
import torch.nn as nn

INIT_CONST = 0.02


class Interpreter(nn.Module):
    def __init__(self,
                universal_action_dim = 128,
                hidden_dim = 256,
                state_dim = 768,
                action_dim = 7):
        super().__init__()
        self.head = Mlp(in_features=state_dim+universal_action_dim, hidden_features=hidden_dim, out_features=action_dim)

    def forward(self, 
                state: torch.Tensor, # B, N, visual_dim
                universal_action: torch.Tensor): # B, UA_dim
        return self.head(torch.cat((state, universal_action), dim = -1)) # B, action_dim


@register_model
def VinillaInterpreter(universal_action_dim, state_dim, **kwargs):
    return Interpreter(
        universal_action_dim = universal_action_dim,
        state_dim = state_dim,
        action_dim = 28
    )









