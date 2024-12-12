import torch
import copy
from model_vit_cnn import Model
from transformers import get_constant_schedule, AutoImageProcessor, AutoModel, get_linear_schedule_with_warmup
import torchvision.models as models


feature_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
vit = AutoModel.from_pretrained("facebook/dinov2-large")

resnet = models.resnet50(pretrained=True)

model = Model(vit, resnet, feature_processor, None)

# List of model files to load
model_files = ["30epo_1e5_dinov2-large-cnn.bin0",
               "30epo_1e5_dinov2-large-cnn.bin4",
               "30epo_1e5_dinov2-large-cnn.bin5",
               "30epo_1e5_dinov2-large-cnn.bin8",
               "30epo_1e5_dinov2-large-cnn.bin9",]

# Load all models
models = [torch.load("./saved_models/checkpoint-best-f1/"+file) for file in model_files]

# Initialize a dictionary to store the averaged weights
averaged_weights = copy.deepcopy(models[0])  # Deepcopy to ensure we do not alter the original

# Average the weights
for key in averaged_weights.keys():
    # Sum the weights for each model
    for i in range(1, len(models)):
        averaged_weights[key] += models[i][key]
    # Compute the average
    averaged_weights[key] = averaged_weights[key] / len(models)

# Load the averaged weights into the model
model.load_state_dict(averaged_weights)

# Save the averaged model to a new .bin file
torch.save(model.state_dict(), "./saved_models/checkpoint-best-f1/averaged_model_cnn.bin")

print("Averaged model saved as 'averaged_model_cnn.bin'.")
