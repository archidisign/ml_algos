import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from sklearn.decomposition import PCA

train_images = np.load('train_images.npy', encoding='latin1')
test_images = np.load('test_images.npy', encoding='latin1')
image1 = (train_images[0][1]).reshape(100,100)
plt.imshow(image1)

def denoise(img):
	labeled, ncomponents = label(img)
	indices = np.indices(img.shape).T[:,:,[1, 0]]
	new_img = np.zeros((100, 100))
	max_arr = [len(indices[labeled==k]) for k in range(1, len(indices))]
	k = np.argmax(max_arr) + 1
	arr = indices[labeled==k]
	i0, j0 = min(arr[:, 0]), min(arr[:, 1])
	imid = int((max(arr[:, 0]) - i0)/2)
	jmid = int((max(arr[:, 1]) - j0)/2)
	for idx in arr:
		i, j = idx
		new_img[i - i0 + 50 - imid, j - j0 + 50 - jmid] = 1
	return new_img

denoised = denoise(image1)
plt.imshow(denoised)


pca = PCA(0.95)
pca.fit(train_images_x)
# pca.fit_transform(train_images_x)
train_images_x = pca.transform(train_images_x)
test_images_x = pca.transform(test_images_x)