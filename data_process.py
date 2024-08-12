import os
from glob import glob
import numpy as np
from shutil import copy

# Data from Kaggle -> LFW - People (Face Recognition)
images_list = glob('./archive/lfw_funneled/**/*.jpg')

# total number of images in the original dataset
print('Total number of images: ', len(images_list))

# path to save sample test images
os.makedirs('./sample_imgs', exist_ok=True)
img_path = './sample_imgs'

# sampling few images for testing
inds = np.arange(len(images_list))
np.random.shuffle(inds)

# copying from original folder
for i in inds[:10]:
    copy(images_list[i], img_path)
