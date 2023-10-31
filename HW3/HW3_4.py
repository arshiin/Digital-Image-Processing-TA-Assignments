import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('Input_data/HeadCT.png',0)

plt.figure(figsize=(20,20))

plt.subplot(3, 2, 1)
plt.title('noisy img')
plt.imshow(img, cmap='gray')

#a
median = cv.medianBlur(img,3)
plt.subplot(3, 2, 2)
plt.title('median')
plt.imshow(median, cmap='gray')

#b
blur = cv.GaussianBlur(median,(5,5),0)
plt.subplot(3, 2, 3)
plt.title('blur')
plt.imshow(blur, cmap='gray')

#c
lapkernel = np.array([[1, 1, 1],
                   [1, -8,1],
                   [1, 1, 1]])
mask = cv.filter2D(blur, -1, lapkernel)

blur = blur - (1)*mask

blur[np.where(blur < 0)] = 0
blur[np.where(blur > 255)] = 255

blur = blur.astype(np.uint8)

print(np.max(blur))
plt.subplot(3, 2, 4)
plt.title('laplacian')
plt.imshow(blur, cmap='gray')

#d
plt.figure(figsize=(20,20))

plt.subplot(3, 2, 1)
plt.title('noisy img')
plt.imshow(img, cmap='gray')

blur = cv.GaussianBlur(img,(5,5),0)
plt.subplot(3, 2, 2)
plt.title('gaussian')
plt.imshow(blur, cmap='gray')

median = cv.medianBlur(blur,3)
plt.subplot(3, 2, 3)
plt.title('median')
plt.imshow(median, cmap='gray')

lapkernel = np.array([[1, 1, 1],
                   [1, -8,1],
                   [1, 1, 1]])
mask = cv.filter2D(blur, -1, lapkernel)

median = median - (1)*mask

median[np.where(blur < 0)] = 0
median[np.where(blur > 255)] = 255

median = median.astype(np.uint8)

print(np.max(median))
plt.subplot(3, 2, 4)
plt.title('laplacian')
plt.imshow(median, cmap='gray')


plt.show()




