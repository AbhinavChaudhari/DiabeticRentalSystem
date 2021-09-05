# -*- coding: utf-8 -*-
"""

"""


import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import cv2
from skimage.io import imshow
plt.gray()
#tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

# === GETTING INPUT SIGNAL

filename = askopenfilename()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread(filename)

plt.imshow(img)
plt.title('ORIGINAL IMAGE')
plt.show()


# PRE-PROCESSING

h1=300
w1=300

dimension = (w1, h1) 
resized_image = cv2.resize(img,(h1,w1))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)

r = resized_image[:,:,0]
g = resized_image[:,:,1]
b = resized_image[:,:,2]


fig = plt.figure()
imshow(r)


#plt.imshow(r)
#plt.title('RED IMAGE')
#plt.show()
#
#plt.imshow(g)
#plt.title('GREEN IMAGE')
#plt.show()
#
#plt.imshow(b)
#plt.title('BLUE IMAGE')
#plt.show()

gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

plt.imshow(gray)
plt.title('GRAY IMAGE')
plt.show()


#==============================================
# IMAGE SEGMENTATION
import cv2
from matplotlib import pyplot as plt

img1 = cv2.medianBlur(resized_image[:,:,1],5)

gray1 = 0.2126 * img1[..., 2] + 0.7152 * img1[..., 1] + 0.0722 * img1[..., 0]

ret,th1 = cv2.threshold(img1,80,100,cv2.THRESH_BINARY)

titles = ['Original Image', 'Initial Segmented Image']

images = [img1, th1]

for i in range(2):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    
plt.show()

#==============================================================================
SPc = np.shape(resized_image)

SEG_im = np.zeros((SPc[0],SPc[1],3))
#
pth ='Dataset\GT\hemorrhages/'
IGT = mpimg.imread(pth+filename[len(filename)-12:len(filename)])
IGT = cv2.resize(IGT,(h1,w1))

for ii in range(0,SPc[0]):
    for jj in range(0,SPc[1]):
        if IGT[ii][jj] == np.max(IGT):
            SEG_im[ii][jj][0] = resized_image[ii,jj,0]+0.75
            SEG_im[ii][jj][1] = resized_image[ii,jj,1]+0.1
            SEG_im[ii][jj][2] = resized_image[ii,jj,2]+0.69
        else:
            SEG_im[ii][jj][0] = resized_image[ii,jj,0]
            SEG_im[ii][jj][1] = resized_image[ii,jj,1]
            SEG_im[ii][jj][2] = resized_image[ii,jj,2]
    
     
plt.imshow(SEG_im)
plt.title('HEMORRHAGES IMAGE')
plt.show()
#==============================================================================


SPc = np.shape(resized_image)

SEG_im = np.zeros((SPc[0],SPc[1],3))
#
pth ='Dataset\GT\hardexudates/'
IGT = mpimg.imread(pth+filename[len(filename)-12:len(filename)])
IGT = cv2.resize(IGT,(h1,w1))

for ii in range(0,SPc[0]):
    for jj in range(0,SPc[1]):
        if IGT[ii][jj] == np.max(IGT):
            SEG_im[ii][jj][0] = resized_image[ii,jj,0]+0.3
            SEG_im[ii][jj][1] = resized_image[ii,jj,1]+0.85
            SEG_im[ii][jj][2] = resized_image[ii,jj,2]+0.66
        else:
            SEG_im[ii][jj][0] = resized_image[ii,jj,0]
            SEG_im[ii][jj][1] = resized_image[ii,jj,1]
            SEG_im[ii][jj][2] = resized_image[ii,jj,2]
    
     
plt.imshow(SEG_im)
plt.title('HARD-EXUDATES IMAGE')
plt.show()
#==============================================================================

SPc = np.shape(resized_image)

SEG_im = np.zeros((SPc[0],SPc[1],3))
#
pth ='Dataset\GT\dots/'
IGT = []
IGT = mpimg.imread(pth+filename[len(filename)-12:len(filename)])
IGT = cv2.resize(IGT,(h1,w1))

for ii in range(0,SPc[0]):
    for jj in range(0,SPc[1]):
        if IGT[ii][jj] == np.max(IGT):
            SEG_im[ii][jj][0] = resized_image[ii,jj,0]+0.9
            SEG_im[ii][jj][1] = resized_image[ii,jj,1]+0.1
            SEG_im[ii][jj][2] = resized_image[ii,jj,2]+0.1
        else:
            SEG_im[ii][jj][0] = resized_image[ii,jj,0]
            SEG_im[ii][jj][1] = resized_image[ii,jj,1]
            SEG_im[ii][jj][2] = resized_image[ii,jj,2]
    
     
plt.imshow(SEG_im)
plt.title('Red-Small DOTS IMAGE')
plt.show()
#==============================================================================


SPc = np.shape(resized_image)

SEG_im = np.zeros((SPc[0],SPc[1],3))
#
pth ='Dataset\GT\softexudates/'
IGT = []
IGT = mpimg.imread(pth+filename[len(filename)-12:len(filename)])
IGT = cv2.resize(IGT,(h1,w1))

for ii in range(0,SPc[0]):
    for jj in range(0,SPc[1]):
        if IGT[ii][jj] != 0:
            SEG_im[ii][jj][0] = resized_image[ii,jj,0]+0.1
            SEG_im[ii][jj][1] = resized_image[ii,jj,1]+0.8
            SEG_im[ii][jj][2] = resized_image[ii,jj,2]+0.9
        else:
            SEG_im[ii][jj][0] = resized_image[ii,jj,0]
            SEG_im[ii][jj][1] = resized_image[ii,jj,1]
            SEG_im[ii][jj][2] = resized_image[ii,jj,2]
    
     
plt.imshow(SEG_im)
plt.title('softexudates IMAGE')
plt.show()
#==============================================================================