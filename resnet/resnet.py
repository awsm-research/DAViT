import torch
import torch.nn as nn

class Model(nn.Module):   
    def __init__(self, resnet, processor, args, num_labels=2):
        super(Model, self).__init__()
        self.resnet = resnet
        self.processor = processor
        self.args = args
        # CLS head
        self.classifier = nn.Linear(1000, num_labels)

    def forward(self, pixel_values, labels=None):
        features = self.resnet(pixel_values)
        logits = self.classifier(features)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            probs = torch.softmax(logits, dim=-1)
            return probs