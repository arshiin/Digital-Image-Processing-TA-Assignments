import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#A


vid = cv.VideoCapture('Input_data/MRI-Head.avi')
frames = []
i = 0
frame_numbers = int(vid.get(cv.CAP_PROP_FRAME_COUNT))  # to get the number of frames
for i in range(frame_numbers):
    ret, frame = vid.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frames.append(gray)
    i += 1
vid.release()

print(len(frames))
first_frame_gray = frames[0]
avg_frame = np.average(frames, axis=0).astype(np.uint8)  # axis=0 since it's a list and we want the rows of it
print(avg_frame.shape)
cv.imwrite('Average Frame.png', avg_frame)

plt.figure()
plt.subplot(1, 3, 1)
plt.title('Noisy')
plt.imshow(first_frame_gray, cmap='gray', vmin=0 , vmax = 255)
plt.axis(False)

plt.subplot(1, 3, 2)
plt.title('Full')
plt.imshow(avg_frame, cmap='gray', vmin=0 , vmax = 255)
plt.axis(False)

#B

avg_frame1 = np.average(frames[:len(frames)//2], axis=0).astype(np.uint8)  # axis=0 since it's a list and we want the rows of it
plt.subplot(1, 3, 3)
plt.title('Half')
plt.imshow(avg_frame1, cmap='gray', vmin=0 , vmax = 255)
plt.axis(False)

# mask1 = np.zeros(avg_frame.shape, 'bool')
# mask2 = np.zeros(avg_frame.shape, 'bool')

# mask1[126:175, 102:147] = True
# mask2[136:179, 35:65] = True

# np.save('mask1', mask1)
# np.save('mask2', mask2)

#C
mask1 = np.load('mask1.npy')
mask2 = np.load('mask2.npy')

print(np.min(mask1))
part1 =np.multiply(avg_frame,mask1)
part2 =np.multiply(avg_frame,mask2)
part3 = cv.add(part1, part2)
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(part1 ,cmap='gray')
plt.axis(False)
plt.subplot(1, 3, 2)
plt.imshow(part2, cmap='gray')
plt.axis(False)
plt.subplot(1, 3, 3)
plt.imshow(part3, cmap='gray')
plt.axis(False)
plt.show()


# In[ ]:




