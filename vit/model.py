import torch
import torch.nn as nn

class Model(nn.Module):   
    def __init__(self, vit, processor, args, num_labels=2):
        super(Model, self).__init__()
        self.vit = vit
        self.processor = processor
        self.args = args
        # CLS head
        self.classifier = nn.Linear(vit.config.hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        hidden_state = self.vit(pixel_values=pixel_values).last_hidden_state
        logits = self.classifier(hidden_state[:, 0, :])
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            probs = torch.softmax(logits, dim=-1)
            return probs