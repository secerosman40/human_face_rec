#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/1 10:49
# @Author  : 周文帆小组
# @Site    : 
# @File    : face_re.py
# @Software: PyCharm

import dlib,os

current_path=os.getcwd()
faces_path=current_path+'/faces/'

options = dlib.shape_predictor_training_options()
options.oversampling_amount=300
options.nu=0.05
options.tree_depth = 2
options.be_verbose=True

train_path=os.path.join(faces_path,'training_with_face_landmarks.xml')
#利用标记好了的xml文件进行人脸特征检测器训练，xml文件在faces文件夹内
dlib.train_shape_predictor(train_path,'predictor.dat',options)
#打印检测器的识别精度
print('\nTraining accuracy:{}'.format(dlib.test_shape_predictor(train_path,'predictor.dat')))



