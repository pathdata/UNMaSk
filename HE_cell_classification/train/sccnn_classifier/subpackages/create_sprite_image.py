import numpy as np
import cv2
from PIL import Image
import os


def create_sprite_image(images, size, sprite_image_save_path):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)

    img_h, img_w = size
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    sprite = np.ones((img_h * n_plots, img_w * n_plots, images.shape[3]))

    for fi in range(n_plots):
        for fj in range(n_plots):
            this_filter = fi * n_plots + fj
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                this_img = cv2.resize(this_img, (size[0], size[1]))
                sprite[fi * img_h:(fi + 1) * img_h, fj * img_w:(fj + 1) * img_w, :] = this_img

    result = Image.fromarray(sprite.astype(np.uint8))
    result.save(os.path.join(sprite_image_save_path, 'sprite_image.png'))
    return sprite
