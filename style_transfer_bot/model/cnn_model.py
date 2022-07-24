from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from style_transfer_bot.config import CNN_MODEL_PARAMS


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CNNNormalization(nn.Module):
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.mean = torch.tensor(mean).to(DEVICE).view(-1, 1, 1)
        self.std = torch.tensor(std).to(DEVICE).view(-1, 1, 1)

    def forward(self, img: torch.tensor):
        return (img - self.mean) / self.std


class StyleLoss(nn.Module):
    def __init__(self, target_feature: torch.tensor):
        super().__init__()
        self.target = self.gram_matrix(target_feature).detach()

    @staticmethod
    def gram_matrix(input_data: torch.tensor):
        b, c, h, w = input_data.size()
        features = input_data.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)
    
    def forward(self, input_data: torch.tensor):
        G = self.gram_matrix(input_data)
        self.loss = F.mse_loss(G, self.target)
        return input_data


class ContentLoss(nn.Module):
    def __init__(self, target: torch.tensor):
        super().__init__()
        self.target = target.detach()

    def forward(self, input_data: torch.tensor):
        self.loss = F.mse_loss(input_data, self.target)
        return input_data


class CNNTransferModel:
    def __init__(self):
        self.content_layers = CNN_MODEL_PARAMS["content_layers_default"]
        self.style_layers = CNN_MODEL_PARAMS["style_layers_default"]
        self.normalization = CNNNormalization(CNN_MODEL_PARAMS["normalization_mean"],
                                              CNN_MODEL_PARAMS["normalization_std"])
        self.cnn_model = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features.to(DEVICE).eval()

    def construct_model(self, style_img: torch.tensor, content_img: torch.tensor):
        content_losses = []
        style_losses = []

        model = nn.Sequential(self.normalization)
        i = 0
        for layer in self.cnn_model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)
            
            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
            
        model = model[:(i + 1)]

        return model, style_losses, content_losses
