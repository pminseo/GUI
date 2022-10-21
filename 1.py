from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# file = '/home/msis/Desktop/my/Digital_Image_Processing/rocket launch.pgm'
file = '/home/msis/Desktop/my/Digital_Image_Processing/images/rocket launch.pgm'
# file = '/home/msis/Desktop/my/Digital_Image_Processing/images/rocket launch2.pgm'

img = Image.open(file)
im = np.array(img)

plt.imshow(im)
plt.show()