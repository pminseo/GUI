
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# 
# # file = '/home/msis/Desktop/my/Digital_Image_Processing/rocket launch.pgm'
# file = '/home/msis/Desktop/my/Digital_Image_Processing/images/rocket launch.pgm'
# # file = '/home/msis/Desktop/my/Digital_Image_Processing/images/rocket launch2.pgm'
# 
# img = Image.open(file)
# im = np.array(img)
# 
# plt.imshow(im)
# plt.show()

import numpy as np

def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    pgm_type =  pgmf.readline()
    if pgm_type ==  b'P5\n':
        len_data = 1
    elif pgm_type ==  b'P2\n':
        pgmf.readline()
        len_data = 4
    else:
        raise TypeError()

    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            raw_dot = pgmf.read(len_data)
            dot = ord(raw_dot) if pgm_type == b'P5\n' else int(raw_dot)
            row.append(dot)
        raster.append(row)
    return raster

with open('./rocket launch.pgm', 'rb') as f:
    resp2 = read_pgm(f)
    resp2 = np.array(resp2)

with open('./rocket launch2.pgm', 'rb') as f:
    resp5 = read_pgm(f)
    resp5 = np.array(resp5)

print(resp2)
print(resp2.shape)
print(resp5.shape)
print(np.all(resp5 == resp2))

