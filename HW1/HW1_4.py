import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math

from sqlalchemy import true

# A
img = cv.imread("Input_data/transformed.png", 0)
plt.figure()
plt.imshow(img, cmap="gray")
plt.title('Original')


# B
for i in range(img.shape[0]):
    if img[i, 0] == 0:
        n = i
        break
for j in range(img.shape[1]):
    if img[0, j] == 0:
        m = j
        break
cy = img.shape[0] / n
cx = img.shape[1] / m
print(cx, cy)
SM = np.float32([[cx, 0, 0], [0, cy, 0]])
scaled = cv.warpAffine(img, SM, (img.shape))

plt.figure()
plt.imshow(scaled, cmap="gray")
plt.title('Scaled')


x = scaled.shape[0]
y = scaled.shape[1]

image = cv.copyMakeBorder(
    scaled, x // 2, y // 2, x // 2, y // 2, cv.BORDER_CONSTANT, None, value=0
)
plt.figure()
plt.imshow(image, cmap="gray")
plt.title('Padded')


# C
x = image.shape[0]
y = image.shape[1]
TM = np.float32([[1, 0, 250], [0, 1, -100]])
translated = cv.warpAffine(image, TM, (x, y))
plt.figure()
plt.imshow(translated, cmap="gray")
plt.title('Translated')


# D
sv = 0
sh = -0.3
SM = np.float32([[1, sh, 0], [sv, 1, 0]])
sheared = cv.warpAffine(image, SM, (x, y))
plt.figure()
plt.imshow(sheared, cmap="gray")
plt.title('Sheared')


# E
ang = -15
ang_rad = np.deg2rad(ang)
M = cv.getRotationMatrix2D((0, 0), ang, 1)
print(M)

forward_image = np.zeros(image.shape, 'uint8')
r, c = image.shape
i, j = np.indices(image.shape)
i_ = np.int32(np.round_(i*np.cos(ang_rad)-j*np.sin(ang_rad)))
j_ = np.int32(np.round_(i*np.sin(ang_rad)+j*np.cos(ang_rad)))
mask = (0<=i_) & (i_<r) & (0<=j_) & (j_<c)
forward_image[i_[mask], j_[mask]] = image[i[mask], j[mask]]

cv.imwrite('forward_image.jpg', forward_image)

            

backward_image = cv.warpAffine(image, M, image.shape[::-1])

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(forward_image, cmap="gray")
plt.title('Rotated Forward')

plt.subplot(1, 2, 2)
plt.title('Rotated Backward')
plt.imshow(backward_image, cmap="gray")
plt.show()


# In[ ]:
