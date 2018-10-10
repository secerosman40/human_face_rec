#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/4 15:03
# @Author  : 周文帆小组
# @Site    : 
# @File    : test_dat.py
# @Software: PyCharm


import os,cv2,dlib,glob


current_path=os.getcwd()
faces_path=current_path+'/faces/'
predictor=dlib.shape_predictor('predictor.dat')
detector=dlib.get_frontal_face_detector()
print('Showing detections and predictions on the images in the faces folder..')

#从数据集导入图片进行模型的测试
for f in glob.glob(os.path.join(faces_path,'*.jpg')):
    print('Processing file : {}'.format(f))
    img=cv2.imread(f)
    #灰色图片处理
    img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dets=detector(img2,1)
    print('Number of faces detected :{}'.format(len(dets)))
    #取得每张脸的坐标值
    for face in dets:
        print('left{};top{};right{};bottom{}'.format(face.left(),face.top(),face.right(),face.bottom()))
        #获取每张脸的特征点
        shape=predictor(img,face)
        #找到每一个特征点所在的图片中的位置，并用圈画出来
        for pt in shape.parts():
            pt_pas=(pt.x,pt.y)
            cv2.circle(img,pt_pas,2,(255,0,0),1)

    cv2.namedWindow(f, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(f,img)

cv2.waitKey(0)
