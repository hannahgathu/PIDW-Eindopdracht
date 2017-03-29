import autograd.numpy as np

file_labels = open('t10k-labels.idx1-ubyte', 'rb')
file_images = open('t10k-images.idx3-ubyte', 'rb')

file_labels.seek(8)
Y = np.fromfile(file_labels, dtype=np.ubyte)

# file_images.seek(0)
# magic = np.fromfile(file_images, dtype=np.int32, count=1)
# num_img = np.fromfile(file_images, dtype=np.int32, count=1)
# num_row = np.fromfile(file_images, dtype=np.int32, count=1)
# num_col = np.fromfile(file_images, dtype=np.uint32, count=1)

file_images.seek(16)
pixels = np.fromfile(file_images, dtype=np.ubyte)
X = np.reshape(pixels, [10000, -1])

file_labels.close()
file_images.close()
