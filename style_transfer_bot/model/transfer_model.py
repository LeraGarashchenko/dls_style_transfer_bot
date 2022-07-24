import io
import math
from typing import Tuple

from PIL import Image
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from style_transfer_bot.model.cnn_model import CNNTransferModel
from style_transfer_bot.utils import get_logger


LOGGER = get_logger("style_transfer_bot")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOGGER.info(f'Device="{DEVICE}"')


class StyleTransfer:
    def __init__(self, img_size: int):
        self.img_size = img_size
        self.image_unloader = transforms.ToPILImage()
        self.cnn_model = CNNTransferModel()

    def load_image(self, image: bytes, img_size_hw: Tuple[int] = None):
        image_loader = transforms.Compose([
            transforms.Resize(img_size_hw if img_size_hw else self.img_size),
            transforms.ToTensor()
        ])

        img = Image.open(io.BytesIO(image))
        img = image_loader(img).unsqueeze(0)
        return img.to(DEVICE)

    def unload_image(self, image: torch.tensor):
        image = image.squeeze(0)
        image = self.image_unloader(image)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()

    def transfer_style(self, max_epochs: int, content_image: bytes, style_image: bytes,
                       style_weight: float, content_weight: float, lr: float):
        content_img = self.load_image(content_image)
        style_img = self.load_image(style_image, tuple(content_img.shape[-2:]))
        input_img = content_img.clone()

        model, style_losses, content_losses = self.cnn_model.construct_model(style_img, content_img)
        LOGGER.debug('Model built')

        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = optim.LBFGS([input_img], lr=lr)

        run = [0]
        scores = [[0]]
        best_score = [(-1, 10000)]
        best_img = [input_img]
        while run[0] < max_epochs:
            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                scores[0].append(loss.item())
                if 0 < scores[0][-1] < best_score[0][1]:
                    best_score[0] = (run[0], scores[0][-1])
                    best_img[0] = input_img.clone().detach()

                if run[0] % 100 == 0:
                    LOGGER.debug(f'{run[0]} / {max_epochs}')
                    LOGGER.debug(f'Style Loss : {style_score:.4f} Content Loss: {content_score:.4f}. '
                                 f'Best score: {best_score[0][1]:.4f}, iter {best_score[0][0]}')

                return style_score + content_score
            
            optimizer.step(closure)
            if math.isnan(scores[0][-1]) or scores[0][-1] / scores[0][-2] > 10:
                LOGGER.debug('Model get worse score. Stop transfer.')
                break

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return self.unload_image(best_img[0])
