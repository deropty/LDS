# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

__all__ = ['ToTensor', 'RandomPatch', 'RandomScale', 'AugMix', ]

import math
import random
from collections import deque

import numpy as np
from PIL import Image

from .functional import to_tensor, augmentations_reid


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 255.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomPatch(object):
    """Random patch data augmentation.
    There is a patch pool that stores randomly extracted pathces from person images.
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1,
                 prob_rotate=0.5, prob_flip_leftright=0.5,
                 ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img


# RandomScale 实现
class RandomScale2(RandomPatch):
    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.75, patch_min_ratio=0.1,
                 scale=None,
                 ):
        super().__init__(prob_happen=prob_happen, pool_capacity=pool_capacity,
                       min_sample_size=min_sample_size, patch_min_area=patch_min_area,
                       patch_max_area=patch_max_area, patch_min_ratio=patch_min_ratio,
                       prob_rotate=0.5, prob_flip_leftright=0.5)
        self.scale = scale
        self.mean= (0.4914, 0.4822, 0.4465)


class RandomScale3(RandomScale2):
    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_ratio=0.1, scale=None,
                 ):
        assert scale is not None, "parameter 'scale' is not assigned."
        self.patch_min_area, self.patch_max_area = scale
        super().__init__(prob_happen=prob_happen, pool_capacity=pool_capacity,
                       min_sample_size=min_sample_size, patch_min_area=self.patch_min_area,
                       patch_max_area=self.patch_max_area, patch_min_ratio=patch_min_ratio,
                       scale=scale)


class RandomScale(RandomScale3):
    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_ratio=0.1, scale=None, threshold=None,
                 ):
        assert scale is not None, "parameter 'scale' is not assigned."
        assert threshold is not None, "parameter 'threshold' is not assigned."
        self.threshold = threshold
        super().__init__(prob_happen=prob_happen, pool_capacity=pool_capacity,
                       min_sample_size=min_sample_size, patch_min_ratio=patch_min_ratio,
                       scale=scale)

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_ratio = random.uniform(self.patch_min_area, self.patch_max_area)
            target_area =  target_ratio * area
            h = int(round(math.sqrt(target_area * H / W)))
            w = int(round(math.sqrt(target_area * W / H)))
            if target_ratio < 1.0:
                if w < W and h < H:
                    return w, h, target_ratio
            else:
                if w > W and h > H:
                    return w, h, target_ratio
        return None, None

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        W, H = img.size  # original image size

        if random.uniform(0, 1) > self.prob_happen:
            return img

        r,g,b = int(self.mean[0]*256), int(self.mean[1]*256), int(self.mean[2]*256)
        patch_resized = Image.new('RGB', (W, H), (r,g,b))


        w, h, ratio = self.generate_wh(W, H)
        img_resized = img.resize((w, h))

        if ratio < self.threshold:
            postion_W, postion_H = (W-w)//2, (H-h)//2
        elif ratio < 1.0:
            postion_W = random.randint(0, W-w)
            postion_H = random.randint(0, H-h)
        else:
            postion_W, postion_H = (w-W)//2, (h-H)//2
            return img_resized.crop((postion_W, postion_H, postion_W+W, postion_H+H))

        patch_resized.paste(img_resized, (postion_W, postion_H))

        return patch_resized


class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    Args:
        aug_prob_coeff: Probability distribution coefficients.
        mixture_width: Number of augmentation chains to mix per augmented example.
        mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
        severity: Severity of underlying augmentation operators (between 1 to 10).
    """

    def __init__(self, aug_prob_coeff=1, mixture_width=3, mixture_depth=-1, severity=1):
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.severity = severity
        self.aug_list = augmentations_reid

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.
        Returns:
          mixed: Augmented and mixed image.
        """
        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        image = np.asarray(image, dtype=np.float32).copy()
        mix = np.zeros_like(image)
        h, w = image.shape[0], image.shape[1]
        for i in range(self.mixture_width):
            image_aug = Image.fromarray(image.copy().astype(np.uint8))
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                image_aug = op(image_aug, self.severity, (w, h))
            mix += ws[i] * np.asarray(image_aug, dtype=np.float32)

        mixed = (1 - m) * image + m * mix
        return mixed

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    from torchvision.transforms import functional as F

    img = Image.open('00.jpg')
    # brightness = 0#(1, 10)
    # contrast = 0#(1, 10)
    # saturation = 0#(1, 10)
    # hue = (0.2, 0.4)
    # transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
    #
    # degrees = 0
    # translate=(0.3, 0)
    # scale=(0.2, 0.6)
    #
    # fillcolor=0 #(int(0.4914* 255), int(0.4822* 255), int(0.4465* 255))
    # transform = transforms.RandomAffine(degrees=degrees, translate=translate,
    #                                     resample=Image.BILINEAR,
    #                                     scale=scale, fillcolor=fillcolor)

    def brightness(im_file):
        # https://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
        im = im_file.convert('L')
        from PIL import Image, ImageStat
        stat = ImageStat.Stat(im)
        return stat.mean[0]

    def contrastness(im_file):
        # https://stackoverflow.com/questions/61658954/measuring-contrast-of-an-image-using-python
        im = im_file.convert('L')
        from PIL import Image, ImageStat
        stat = ImageStat.Stat(im)
        for band, name in enumerate(im.getbands()):
            print(f'Band: {name}, min/max: {stat.extrema[band]}, stddev: {stat.stddev[band]}')
        return stat.stddev[0]

    # print(contrastness(img))
    # img = F.adjust_contrast(img, 2)
    # print(contrastness(img))
    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()
    # trans = transforms.Resize([256, 128], interpolation=3); img = trans(img)
    # trans = transforms.RandomHorizontalFlip(p=1); img = trans(img)
    # trans = transforms.Pad(10, padding_mode='constant'); img = trans(img)
    # trans = transforms.RandomCrop([256, 128]); img = trans(img)
    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()
    # trans = RandomScale4(prob_happen=1, scale=(0.5, 0.9)); img2 = trans(img)
    # plt.axis('off')
    # plt.imshow(img2)
    # plt.show()
    # trans = transforms.ToTensor(); img = trans(img)
    # trans = transforms.RandomErasing(p=1, value=[0.485 * 255, 0.456 * 255, 0.406 * 255]); img = trans(img)
    # trans = transforms.ToPILImage(); img = trans(img)
    # plt.axis('off')
    # plt.imshow(img)

    # plt.axis('off')
    # plt.imshow(img)
    # plt.show()
    trans = RandomScale(prob_happen=1, scale=(0.3, 1.5), threshold=0.7)
    img = trans(img)
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    # transform = RandomPatch(prob_happen=1)
    # for attempt in range(99):
    #     trans_img = transform(img)
    # trans_img = transform(img)
    # plt.imshow(trans_img)
    # plt.show()