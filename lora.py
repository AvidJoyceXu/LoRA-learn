import torch
import transformers

from utils import recursive_getattr, recursive_setattr


class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling):
        super(LoRALinear, self).__init__()
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        # TODO: Implement lora left and right weights
        self.lora_left_weight = torch.nn.Parameter(torch.zeros(weight.size(0), lora_dim)) # B
        self.lora_right_weight = torch.nn.Parameter(torch.zeros(lora_dim, weight.size(1))) # A
        #############################################
        self.lora_scaling = lora_scaling / lora_dim
        self.init_parameters()
        # TODO: Freeze original weight and bias
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        #######################################

    def init_parameters(self):
        # TODO: Initialize LoRA parameters
        self.lora_right_weight.data.normal_(0, 0.02) # Use random Gaussian initialization for A
        # Initialize B with zero
        ##################################

    def forward(self, input):
        # TODO: Implement the forward function
        # W = W + BA
        return torch.nn.functional.linear(input, self.weight + self.lora_scaling * torch.mm(self.lora_left_weight, self.lora_right_weight), self.bias)
        ######################################


def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D)) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            tmp = LoRALinear(module.weight.t().detach(), module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    return model


def only_optimize_lora_parameters(model: torch.nn.Module):
    # TODO: Turn off the gradient of all the parameters except the LoRA parameters
    for name, module in model.named_parameters():
        if 'lora' not in name:
            module.requires_grad = False
    return model
    ##############################################################################

def get_lora_state_dict(model: torch.nn.Module):
    # TODO: return lora left and right weights as state dict
    # The saved state dict will be used later for loading
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state_dict[name + '.lora_left_weight'] = module.lora_left_weight
            state_dict[name + '.lora_right_weight'] = module.lora_right_weight
    return state_dict
    ########################################################