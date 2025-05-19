import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    def __init__(self, in_channels, num_hidden_1, num_hidden_2, num_hidden_3, num_classes):
        super(CNNBaseline, self).__init__()
        self.in_channels = in_channels
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.num_hidden_3 = num_hidden_3
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            # Conv Block 1
            # Input: INPUT_CHANNELS*224*224
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_hidden_1, kernel_size=3, padding=1),  # -> (h1, 224, 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> (h1, 112, 112)
            nn.Dropout(p=0.25),

            # Conv Block 2
            nn.Conv2d(self.num_hidden_1, self.num_hidden_2, kernel_size=3, padding=1),  # -> (h2, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> (h2, 56, 56)
            nn.Dropout(p=0.25),

            # Conv Block 3
            nn.Conv2d(self.num_hidden_2, self.num_hidden_3, kernel_size=3, padding=1),  # -> (h3, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> (h3, 28, 28)
            nn.Dropout(p=0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # -> (h3 * 28 * 28)
            nn.Linear(self.num_hidden_3 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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


def get_resnet18_classifier(num_classes, pretrained = True, freeze_backbone = True, dropout = 0.0, print_architecture = False):
    """
    Initialize a ResNet-18 Classifier Model

    Args:
        num_classes: Number of classes for the final classification layer
        pretrained: Load a pretraiend model. Defaults to True.
        freeze_backbone: Freeze convolution layers during training. Defaults to False.
        dropout: Add dropout layers after each ReLU activation with drop probability. Defaults to 0. (Do Not Add)
        print_architecture: Prints out the model architecture. Defaults to False.
    Returns:
        model: A ResNet-18 classifier model
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


def get_resnet50_classifier(num_classes, pretrained = True, freeze_backbone = True, dropout = 0.0, print_architecture = False):
    """
    Initialize a ResNet-50 Classifier Model

    Args:
        num_classes: Number of classes for the final classification layer
        pretrained: Load a pretraiend model. Defaults to True.
        freeze_backbone: Freeze convolution layers during training. Defaults to False.
        print_architecture: Prints out the model architecture. Defaults to False.
    Returns:
        model: A ResNet-50 classifier model
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
