from models.modeling_bert import BertOutput
from typing import List, Optional, Tuple
from torch import nn
from mamba_ssm import Mamba
import torch

class Identity(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        return (hidden_states, )
            
class MambaBlock(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.args_mamba = args.mamba
        self.mamba_dstate = args.mamba_dstate

        print(f"Using Mamba with d_state: {self.mamba_dstate}")
        self.mamba = Mamba(d_model=768, # Model dimension d_model
                                d_state=self.mamba_dstate,  # SSM state expansion factor
                                d_conv=4,    # Local convolution width
                                expand=2,    # Block expansion factor
                                )
        
        if "BD" in self.args_mamba:

            print(f"Using Mamba with Bi-Directionality")

            self.mamba_rev = Mamba(d_model=768, # Model dimension d_model
                                    d_state=4,  # SSM state expansion factor
                                    d_conv=4,    # Local convolution width
                                    expand=2,    # Block expansion factor
                                    )
            
        if "outFFN" in self.args_mamba:

            print(f"Using Mamba with Additioinal FFN Layer")

            self.output = BertOutput(config)

    
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        if "BD" in self.args_mamba:
            out = self.mamba(hidden_states,) + self.mamba_rev(hidden_states.flip(dims=[1])).flip(dims=[1], )
        else:
            out = self.mamba(hidden_states,)
                
        if "outFFN" in self.args_mamba:
            out = self.output(out, hidden_states)

        return (out, )
    
def model_surgery_for_mamba(args, original_model):
    if args.model_name_or_path == 'bert-base-uncased':
        config = original_model.bert.encoder.config
        for i in range(12):
            if i < args.mamba_layers:
                original_model.bert.encoder.layer[i].attention = MambaBlock(config, args)
            else:
                original_model.bert.encoder.layer[i].attention = Identity(config)

    return original_model