#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/4 15:44
# @Author  : 周文帆小组
# @Site    : 
# @File    : face_recoginition.py
# @Software: PyCharm


import sys,os,dlib,cv2,glob
#将每一张人脸扩展到128维特征空间,并计算相应的向量值
#dlib_face_recogition_mod.dat是官方提供的特征空间向量计算模型

#最终的结果文件是在face_compare这个py文件中

current_path=os.getcwd()
#特征检测器路径
#获取向量计算器路径
predictor_path=current_path+'/predictor.dat'
face_rec_model_path=current_path+'/dlib_face_recognition_mod.dat'
#获取图片路径
faces_folder_path=current_path+'/faces/'

detector=dlib.get_frontal_face_detector()
shape_pre=dlib.shape_predictor(predictor_path)
face_model=dlib.face_recognition_model_v1(face_rec_model_path)

for img_path in glob.glob(os.path.join(faces_folder_path,'*.jpg')):
    print(img_path)
    #载入图片的色彩
    img=cv2.imread(img_path,cv2.IMREAD_COLOR)
    #格式转换
    b,g,r=cv2.split(img)
    img2=cv2.merge([r,g,b])

    faces=detector(img,1)
    print('there is {} faces'.format(len(faces)))

    #每一张脸的特征点
    for index,face in enumerate(faces):
        shape=shape_pre(img2,face)

        for p in shape.parts():
            p_pots=(p.x,p.y)
            cv2.circle(img,p_pots,2,(0,255,0),1)
        #显示相应的特征圈圈
        cv2.namedWindow(img_path+str(index),cv2.WINDOW_AUTOSIZE)
        cv2.imshow(img_path+str(index),img)

        #计算每一张脸的特征点的128维特征向量,并输出
        face_description=face_model.compute_face_descriptor(img2,shape)
        print(face_description)

k=cv2.waitKey(0)
cv2.destroyAllWindows()


