import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from fastreid.data.transforms.transforms import RandomPatch, RandomScale
img = Image.open('tina.jpg')
# brightness = (1, 10)
# contrast = (1, 10)
# saturation = (1, 10)
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

plt.imshow(img)
plt.show()
transform = RandomScale(prob_happen=1, scale=(0.2, 0.8))
for attempt in range(99):
    trans_img = transform(img)
trans_img = transform(img)
plt.imshow(trans_img)
plt.show()

