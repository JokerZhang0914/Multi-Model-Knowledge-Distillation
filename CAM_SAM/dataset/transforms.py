import numpy as np
import random
from PIL import Image, ImageFilter, ImageOps

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)

    proc_img[..., 0] = (imgarr[..., 0] - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] - mean[2]) / std[2]
    return proc_img


def normalize_img2(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)

    proc_img[..., 0] = (imgarr[..., 0] - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] - mean[2]) / std[2]
    return proc_img

def _img_rescaling(image, label=None, scale=None):
    
    #scale = random.uniform(scales)
    h, w, _ = image.shape
    
    new_scale = [int(scale * w), int(scale * h)]

    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_image = np.asarray(new_image).astype(np.float32)

    if label is None:
        return new_image

    new_label = Image.fromarray(label).resize(new_scale, resample=Image.NEAREST)
    new_label = np.asarray(new_label)
    
    return new_image, new_label

def random_scaling(image, label=None, scale_range=None):

    min_ratio, max_ratio = scale_range
    assert min_ratio <= max_ratio

    ratio = random.uniform(min_ratio, max_ratio)

    return _img_rescaling(image, label, scale=ratio)

def random_fliplr(image, label=None):
    p = random.random()

    if label is None:
        if p > 0.5:
            image  = np.fliplr(image)
        return image
    else:
        if p > 0.5:
    
            image = np.fliplr(image)
            label = np.fliplr(label)

        return image, label
    
def random_crop(image, label=None, crop_size=None, mean_rgb=[0,0,0], ignore_index=255):

    h, w, _ = image.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_image = np.zeros((H,W,3), dtype=np.uint8)
    
    pad_image[:,:,0] = mean_rgb[0]
    pad_image[:,:,1] = mean_rgb[1]
    pad_image[:,:,2] = mean_rgb[2]
    
    H_pad = int(np.random.randint(H-h+1))
    W_pad = int(np.random.randint(W-w+1))
    
    pad_image[H_pad:(H_pad+h), W_pad:(W_pad+w), :] = image
    
    def get_random_cropbox(_label, cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            if _label is None:
                return H_start, H_end, W_start, W_end, 

            temp_label = _label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]

            if len(cnt>1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end, 

    H_start, H_end, W_start, W_end = get_random_cropbox(label)

    crop_image = pad_image[H_start:H_end, W_start:W_end,:]

    img_H_start = max(H_pad-H_start, 0)
    img_W_start = max(W_pad-W_start, 0)
    img_H_end = min(crop_size, h + H_pad - H_start)
    img_W_end = min(crop_size, w + W_pad - W_start)
    img_box = np.asarray([img_H_start, img_H_end, img_W_start, img_W_end], dtype=np.int16)

    if label is None:

        return crop_image, img_box

    pad_label = np.ones((H,W), dtype=np.uint8) * ignore_index
    pad_label[H_pad:(H_pad+h), W_pad:(W_pad+w)] = label
    label = pad_label[H_start:H_end, W_start:W_end]

    return crop_image, label, img_box   