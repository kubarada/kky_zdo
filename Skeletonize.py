from skimage.morphology import skeletonize
from skimage import data, io
import matplotlib.pyplot as plt
from skimage.util import invert

PATH_TO_IMG = 'cvat_dataset/images/default/SA_20211012-164802_incision_crop_0.jpg'

# Invert the horse image
image = io.imread(PATH_TO_IMG)
image = invert(image)

# perform skeletonization
skeleton = skeletonize(image)

io.imshow(skeleton, cmap=plt.cm.gray)
plt.show()
io.imshow(image, cmap=plt.cm.gray)
plt.show()