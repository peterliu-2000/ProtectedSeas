import torch
import torchvision.models as models
import torch.nn as nn

# The following thing silences the ssl error when loading pre-trained model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def append_dropout(model, rate):
        for name, module in model.named_children():
            # We only add dropout layers at the end of each resnet layer
            if ("layer" in name):
                resnet_block_layers = list(module)
                resnet_block_layers.append(nn.Dropout(p=rate)) # MPS doesnt support inplace dropout.
                new_block = nn.Sequential(*resnet_block_layers)
                setattr(model, name, new_block)


def get_resnet18_classifier(num_classes, pretrained = True, freeze_backbone = False, dropout = 0.0, print_architecture = False):
    """
    Initialize a ResNet-18 Classifier Model

    Args:
        num_classes: Number of classes for the final classification layer
        pretrained: Load a pretraiend model. Defaults to True.
        freeze_backbone: Freeze convolution layers during training. Defaults to False.
        dropout: Add dropout layers after each ReLU activation with drop probability. Defaults to 0. (Do Not Add)
        print_architecture: Prints out the model architecture. Defaults to False.
    """
    if pretrained:
        model = models.resnet18(weights = "IMAGENET1K_V1")
    else:
        model = models.resnet18()
        
    if pretrained and freeze_backbone:
        for name, layer in model.named_children():
            if name in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    # Resets the final classification layer
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        
    # Insert Dropout Layers to the model if specified
    if dropout > 0.0:
        append_dropout(model, dropout)
        
    if print_architecture:
        print(model)
    
    return model


def get_resnet50_classifier(num_classes, pretrained = True, freeze_backbone = False, dropout = 0.0, print_architecture = False):
    """
    Initialize a ResNet-50 Classifier Model

    Args:
        num_classes: Number of classes for the final classification layer
        pretrained: Load a pretraiend model. Defaults to True.
        freeze_backbone: Freeze convolution layers during training. Defaults to False.
        print_architecture: Prints out the model architecture. Defaults to False.
    """
    if pretrained:
        model = models.resnet50(weights = "IMAGENET1K_V2")
    else:
        model = models.resnet50()
        
    if pretrained and freeze_backbone:
        for name, layer in model.named_children():
            if name in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]:
                for param in layer.parameters():
                    param.requires_grad = False
    
    # Resets the final classification layer
    model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    
    if dropout > 0.0:
        append_dropout(model, dropout)
    
    if print_architecture:
        print(model)
        
    return model