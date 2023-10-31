import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pydicom

#A
img = pydicom.dcmread('Input_data/file1.dcm')

#B
b0 = img.BitsStored
print(img.BitsAllocated )
print(img.BitsStored)
print(img.Modality)
l = 2**(img.BitsStored) - 1
print(img.BodyPartExamined )

#C
img = img.pixel_array
print(img.shape)
x,y = img.shape
img1 = img[::4]
img2 = img[:, ::4]

plt.figure()
plt.subplot(1, 4, 1)
plt.title('img')
plt.imshow(img, cmap='gray', vmin=0, vmax=l)
plt.xlim([0, x])
plt.ylim([y, 0]) 

plt.subplot(1, 4, 2)
plt.title('img1')
plt.imshow(img1, cmap='gray', vmin=0, vmax=l)
plt.xlim([0, x])
plt.ylim([y, 0]) 


plt.subplot(1, 4, 3)
plt.title('img2')
plt.imshow(img2, cmap='gray', vmin=0, vmax=l)
plt.xlim([0, x])
plt.ylim([y, 0]) 


#D
img3 = img[::2, ::2]

plt.subplot(1, 4, 4)
plt.title('img3')
plt.imshow(img3, cmap='gray', vmin=0, vmax=l)
plt.xlim([0, x])
plt.ylim([y, 0]) 
  
plt.show()

#F
img_stretch1=cv.resize(img3,(0,0),fx=2,fy=2,interpolation=cv.INTER_NEAREST)
img_stretch2=cv.resize(img3,(0,0),fx=2,fy=2,interpolation=cv.INTER_LINEAR)
img_stretch3=cv.resize(img3,(0,0),fx=2,fy=2,interpolation=cv.INTER_CUBIC)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(img_stretch1, cmap='gray', vmin=0, vmax=l)
plt.xlim([0, x])
plt.ylim([y, 0]) 
  
plt.subplot(1, 3, 2)
plt.imshow(img_stretch2, cmap='gray', vmin=0, vmax=l)
plt.xlim([0, x])
plt.ylim([y, 0]) 
  
plt.subplot(1, 3, 3)
plt.imshow(img_stretch3, cmap='gray', vmin=0, vmax=l)
plt.xlim([0, x])
plt.ylim([y, 0])
plt.show()

  
#G
print(np.max(img))

def func(img, reducing_bits):
    return (img//2**reducing_bits)



plt.figure()
plt.subplot(2, 3, 1)
plt.title('8 bits')
plt.imshow(func(img, b0-8), cmap='gray', vmin=0 , vmax = 255)
plt.axis(False)

plt.subplot(2, 3, 2)
plt.title('5 bits')
plt.imshow(func(img, b0-5), cmap='gray', vmin=0 , vmax = 31)
plt.axis(False)

plt.subplot(2, 3, 3)
plt.title('3 bits')
plt.imshow(func(img, b0-3), cmap='gray', vmin=0 , vmax =7)
plt.axis(False)

plt.subplot(2, 3, 4)
plt.title('2 bits')
plt.imshow(func(img, b0-2), cmap='gray', vmin=0 , vmax =3)

plt.axis(False)

plt.subplot(2, 3, 5)
plt.title('1 bits')
plt.imshow(func(img, b0-1), cmap='gray' ,vmin=0 , vmax =1)
plt.axis(False)
print(np.max(func(img,1)))
plt.show()

#H
img_final = func(img, 8)
cv.imwrite('img_final.tif', img_final)
cv.imwrite('img_final1.bmp',img_final)



