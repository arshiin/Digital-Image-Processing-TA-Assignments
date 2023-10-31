import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# A, B, C
intersection = lambda img1, img2: np.min((img1, img2), 0)
difference = lambda img1, img2: img1 - intersection(img1, img2)


def save(img, s):
    img[img < 127.5] = 0
    img[img >= 127.5] = 255
    cv.imwrite(s, img)


img = cv.imread('Input_data/4_1.png', 0)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
kernel_1 = cv.getStructuringElement(cv.MORPH_RECT, (70, 70))

img_1 = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_1)

img_2 = img_1 - cv.erode(img_1, kernel)

img_3 = cv.imread('Input_data/chopper.png', 0)
# img_4 = img_1 - img_3
img_4 = difference(img_1, img_3)
print(np.any(img_4 == 1))

for x in (img, img_1, img_2, img_3, img_4):
    cv.imshow('win', x)
    cv.waitKey(1000)

# D
img = cv.imread('Input_data/Blobs.png', 0)

plt.imshow(img, cmap='gray', vmin=0, vmax=255)

# E
kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (55, 55))

closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel1)

kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (95, 95))

opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel1)

plt.figure(figsize=(20, 20))

plt.subplot(2, 2, 1)
plt.title('texture segmentation')
plt.imshow(closing, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imshow(opening, cmap='gray', vmin=0, vmax=255)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()

# F
img = 255 - img
array = np.zeros(20, dtype='float32')
k = 0
for i in range(5, 105, 5):
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (i, i))
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel1)
    array[k] = np.sum(img[:, :])
    # print(array[k])
    k = k + 1
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()
print(i)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.xlabel("r")
plt.ylabel("differences in surface area")

x = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
# x = np.zeros(151, dtype='float64')
# for i in range (1,152):
# x[i-1] = i

# print(x[150])
plt.plot(x, array)
plt.show()
print(array)
array1 = -1 * np.diff(array)
print(array1)
array[0:19] = array1
array[19:20] = 0
plt.plot(x, array)
plt.show()
