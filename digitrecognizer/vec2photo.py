import cv2
import pandas as pd
import os

Image_width = 28
Image_heigh = 28
image_path = "G:/数据集/digit_photos/"
i = 0
data = pd.read_csv("G:/Kaggle/DigitRecognizer/train.csv")
train_data = data.iloc[:, 1:].values
label_data = data.iloc[:, [0]].values.ravel()

# 建立存放图片文件的文件夹
if not os.path.exists(image_path):
    os.makedirs(image_path)
# 将图片存放的相应文件夹
for lab in label_data:
    # 创建分类的图片文件夹
    imgs_path = os.path.join(image_path, str(lab))
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)

    # 读取文件建立图片,并放到相应的文件夹中
    img_data = train_data[i]
    img = img_data.reshape([Image_width, Image_heigh, 1])
    img_dir = imgs_path + '/' + str(i) + '.jpg'
    # print 'img_dir', img_dir
    cv2.imencode('.jpg', img)[1].tofile(img_dir)
    if (i + 1) % 100 == 0:
        print('deal the %s' % i)
    i = i + 1