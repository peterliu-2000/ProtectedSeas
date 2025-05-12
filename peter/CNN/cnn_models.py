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



