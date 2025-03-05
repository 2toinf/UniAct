from timm.models.registry import register_model
from timm.models.layers import Mlp
import torch
import torch.nn as nn
import math
INIT_CONST = 0.02

def get_positional_embeddings(seq_length, d_model):
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


class ACT_Decoder(nn.Module):
    def __init__(self,
                universal_action_dim = 128,
                hidden_dim = 512,
                proprio_dim = 14,
                action_dim = 14,
                input_length = 49*3+2,
                action_chunking_length = 4):
        super().__init__()
        
        self.action_chunking_length = action_chunking_length
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.ua_proj = nn.Linear(universal_action_dim, hidden_dim)
        
        
        self.action_head = nn.Sequential(*[
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, action_dim)])
        
        self.queries = nn.Parameter(torch.zeros(1, action_chunking_length, hidden_dim), requires_grad=True)
        self.queries_pos_emb = nn.Parameter(get_positional_embeddings(action_chunking_length, hidden_dim), requires_grad=False)
        self.input_pos_emb = nn.Parameter(get_positional_embeddings(input_length, hidden_dim), requires_grad=False)
        
        
        assert hidden_dim % 64 == 0
        self.model = nn.Transformer(
            d_model=hidden_dim,
            nhead=hidden_dim//64,
            num_encoder_layers = 4,
            num_decoder_layers = 2,
            dim_feedforward = hidden_dim * 4,
            dropout = 0.0,
            batch_first = True,
            norm_first = False,
        )
        

    def forward(self, 
                vision_embedding: torch.Tensor,  # B V N C
                universal_action: torch.Tensor, # B, ua_dim
                proprios: torch.Tensor): # B, prio_dim
        B = vision_embedding.shape[0]
        inputs = torch.cat(
            [
                vision_embedding.flatten(start_dim=1, end_dim=2),
                self.ua_proj(universal_action).unsqueeze(1),
                self.proprio_proj(proprios).unsqueeze(1)
            ], dim = 1
        )
        inputs = inputs + self.input_pos_emb
        query = self.queries.repeat(B, 1, 1) + self.queries_pos_emb
        
        output = self.model.forward(inputs, query) # B ac hidden
        output = self.action_head(output) # B ac 14
        return output
        
        
@register_model
def ACT_2RGB_7DoFs_9Proprio_4FutureAction(**kwargs):
    return ACT_Decoder(proprio_dim = 9,
                action_dim = 7,
                input_length = 49 * 2 + 2)

@register_model
def ACT_2RGB_7DoFs_7Proprio_4FutureAction(**kwargs):
    return ACT_Decoder(proprio_dim = 7,
                action_dim = 7,
                input_length = 49 * 2 + 2)


@register_model
def ACT_3RGB_14DoFs_14Proprio_4FutureAction(**kwargs):
    return ACT_Decoder(proprio_dim = 14,
                action_dim = 14,
                input_length = 49*3+2)









