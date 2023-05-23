from skimage.morphology import skeletonize
from skimage import data, io
import matplotlib.pyplot as plt
from skimage.util import invert
import cv2

PATH_TO_IMG = '../cvat_dataset/images/default/SA_20211012-164802_incision_crop_0.jpg'

# Invert the horse image
image = io.imread(PATH_TO_IMG)
image = invert(image)

# perform skeletonization

skeleton = skeletonize(image)
cv2.imwrite('skeleton.png', skeleton)
cv2.imwrite('ioRead.png', image)

