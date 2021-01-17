import cv2
import numpy as np

#img = cv2.imread('C:/Users/cxt/Desktop/TXT/chess1.bmp').astype(np.uint8)
img = cv2.imread('E:/Image/Left/L_1.jpg')
cv2.imshow('img',img)
#imgu = cv2.imread('C:/Users/cxt/Desktop/TXT/Truechess1.bmp').astype(np.uint8)
imgu = cv2.imread('E:/Image/Left/true1.jpg')

imgu = cv2.resize(imgu,(1440,1080),interpolation=cv2.INTER_CUBIC)
cv2.imshow('xc',imgu)
print(img.shape)
print(imgu.shape)

err = cv2.absdiff(img,imgu)
#err1 = np.abs(img - imgu)  #差值的绝对值
cv2.imshow('err',err)
import matplotlib.pyplot as plt
fig = plt.figure('result')
plt.axis('off')  #关闭坐标轴
plt.subplot(2,2,1)     #将窗口分为两行两列四个子图，则可显示四幅图片
plt.title('imgsrc')   #第一幅图片标题
plt.imshow(img)
'''
plt.subplot(2,2,2)
plt.title('imgupsampling')
plt.imshow(err1)
'''
plt.subplot(2,2,3)
plt.title('imgupsampling')
plt.imshow(imgu)

plt.subplot(2,2,4)
plt.title('imgerr')
plt.imshow(err)

fig.tight_layout()#调整整体空白
plt.subplots_adjust(wspace =0)#调整子图间距
plt.show()   #显示