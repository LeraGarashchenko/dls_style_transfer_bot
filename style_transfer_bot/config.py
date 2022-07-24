import os

TOKEN = os.getenv("TOKEN")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

TRANSFER_PARAMS = {
    "img_size": 512,
    "n_epochs": 600,
    "style_weight": 10000,
    "content_weight": 0.01,
    "lr": 0.1
}

CNN_MODEL_PARAMS = {
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "content_layers_default": ['conv_4'],
    "style_layers_default": ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
}
