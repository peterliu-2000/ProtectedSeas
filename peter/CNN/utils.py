import torch.nn as nn

def append_dropout(model, rate):
        for name, module in model.named_children():
            # We only add dropout layers at the end of each resnet layer
            if ("layer" in name):
                resnet_block_layers = list(module)
                resnet_block_layers.append(nn.Dropout(p=rate)) # MPS doesnt support inplace dropout.
                new_block = nn.Sequential(*resnet_block_layers)
                setattr(model, name, new_block)