import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Second Sequential Block
        self.conv1 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(2048)
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(2048)
        self.conv3 = nn.Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Second Sequential Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class Model(nn.Module):   
    def __init__(self, vit, processor, args, num_labels=2):
        super(Model, self).__init__()
        self.vit = vit
        self.cnn = CNN()
        
        self.processor = processor
        self.args = args
        
        # CLS head
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # Fully connected layer for 1000 features (from ResNet)
        self.fc = nn.Linear(2048, 1000)  # +1024
        
        # Classifier for binary classification (num_classes=2 by default)
        self.classifier = nn.Linear(1000, num_labels)
                
    def forward(self, pixel_values, labels=None):
        hidden_state = self.vit(pixel_values=pixel_values).last_hidden_state
        hidden_state = hidden_state[:, 1:, :] # ignore special token
        batch_size, w_h, channels  = hidden_state.shape
        w_h = int(math.sqrt(w_h))
        hidden_state = hidden_state.view(batch_size, w_h, w_h, channels)
        hidden_state = hidden_state.permute(0, 3, 1, 2)
        
        hidden_state = self.cnn(hidden_state)
        
        hidden_state = self.avgpool(hidden_state)
        hidden_state = torch.flatten(hidden_state, 1)  # Flatten to feed into the fully connected layers
                
        hidden_state = self.fc(hidden_state)
        logits = self.classifier(hidden_state)
                
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            probs = torch.softmax(logits, dim=-1)
            return probs