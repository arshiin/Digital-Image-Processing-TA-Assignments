import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Input_data/MRI.png', 0)

def func(img , filter_name):
    
    if filter_name == 'averaging':
        avg = cv.blur(img , (3,3))
        return avg
    
    if filter_name == 'median': 
        med = cv.medianBlur(img ,3)
        return med
        
    if filter_name =='sobel_x': 
        sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)  # x
        return sobelx
    
    if filter_name =='sobel_y': 
        sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)  # y
        return sobely
        
    if filter_name == 'laplacian':
        lapkernel = np.array([[0, -1, 0],
                   [-1, 4,-1],
                   [0, -1, 0]])
        mask = cv.filter2D(img, -1, lapkernel)
        return mask
        
img = cv.imread('Input_data/MRI.png', 0)      

fig = plt.figure(figsize = (10,10))

plt.subplot(3, 2, 1)
plt.title('original')
plt.imshow(img, cmap='gray')

plt.subplot(3, 2, 2)
plt.title('averaging(3)')
plt.imshow(func(img, 'averaging'), cmap='gray')

plt.subplot(3, 2, 3)
plt.title('Median (3)')
plt.imshow(func(img, 'median'), cmap='gray')

plt.subplot(3, 2, 4)
plt.title('sobelx')
plt.imshow(func(img, 'sobel_x'), cmap='gray', vmin=0, vmax=255)

plt.subplot(3, 2, 5)
plt.title('Sobely')
plt.imshow(func(img, 'sobel_y') , cmap='gray', vmin=0, vmax=255)

plt.subplot(3, 2, 6)
plt.title('laplacain')
plt.imshow(func(img, 'laplacian'), cmap='gray', vmin=0, vmax=255)
plt.show()

med = cv.medianBlur(img ,5)
avg = cv.blur(img , (5,5))

fig = plt.figure(figsize = (10,10))

plt.subplot(1, 2, 1)
plt.title('averaging 5')
plt.imshow(avg , cmap='gray')

plt.subplot(1, 2, 2)
plt.title('median 5')
plt.imshow(med, cmap='gray')

plt.show()

kernel = np.array([[1, 4, 1],
                   [4, 16,4],
                   [1, 4, 1]])
kernel = kernel / 36
out = cv.filter2D(img, -1, kernel)
print(np.max(out))
plt.imshow(out , cmap = 'gray')





