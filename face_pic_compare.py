#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/4 16:38
# @Author  : 周文帆小组
# @Site    : 
# @File    : face_ensure.py
# @Software: PyCharm


import os,dlib,glob,cv2,re
import numpy as np


#依据欧氏距离，来判断样本的类别
def compare_face(face1,face2):
    if face1==[] or face2==[]:
        dis=100
    else:
        dis=0
        for i in range(len(face1)):
            dis+= (face1[i]-face2[i])**2
        dis=np.sqrt(dis)
        if dis<0.55:
            print(dis)
    return dis

#创建单个人脸照片的实例类,并对每一个人脸计算在128维的特征空间向量
class Face_rec(object):
    def __init__(self):

        self.current_path=os.getcwd()
        self.predictor_path=self.current_path+'/predictor.dat'
        self.face_rec_model_path=self.current_path+'/dlib_face_recognition_mod.dat'
        self.faces_folder_path=self.current_path+'/faces/'
        self.data_path=self.current_path+'/data/'

        self.detector=dlib.get_frontal_face_detector()
        self.face_rec_model=dlib.face_recognition_model_v1(self.face_rec_model_path)

        self.name=None
        self.img_bgr=None
        self.img_rgb=None

        self.detector=dlib.get_frontal_face_detector()
        self.shape_predictor=dlib.shape_predictor(self.predictor_path)
    #输入一个图片并修改色调
    def inputPerson(self,img_path=None):
        self.path=img_path
        if self.path==None:
            print('No such pic')
            return
        #形成新的颜色图片
        self.img_bgr=cv2.imread(self.path)
        b,g,r=cv2.split(self.img_bgr)
        self.img_rgb=cv2.merge([r,g,b])
    #计算128维度空间向量
    def create_128(self):
        face_dis=[]
        faces=self.detector(self.img_rgb,1)
        if len(faces)==0:
            face_dis=[]
            print('此照片无人脸')
        else:
            print('照片中人脸数有{}'.format(len(faces)))
            for face in faces:
                print('left{},right{},top{},bottom{}'.format(face.left(),face.right(),face.top(),face.bottom()))
                shape=self.shape_predictor(self.img_rgb,face)
                face_dis=self.face_rec_model.compute_face_descriptor(self.img_rgb,shape)

        return face_dis

#到unknown照片集中进行识别，并依次输出这里面的张照片者的名字
#最终执行程序
def main():
    vec,names,unknown_vec,unknown_names=[],[],[],[]
    face_rec=Face_rec()

    current_path=os.getcwd()
    img_path=current_path+'/human_faces/'
    pattern=re.compile('human_faces/(.*?).jpg')
    pattern_unknown=re.compile('unknown/(.*?).jpg')
    #读取每一个数据集中的人照片
    for path in glob.glob(os.path.join(img_path,'*.jpg')):
        name=re.findall(pattern,path)[0]
        face_rec.inputPerson(img_path=path)
        vec1=face_rec.create_128()
        vec.append(vec1)
        names.append(name)
    #读取每一个需要识别的照片人脸，并计算128向量
    unknown_path=current_path+'/unknown/'
    for path in glob.glob(os.path.join(unknown_path, '*.jpg')):
        name = re.findall(pattern_unknown, path)[0]
        face_rec.inputPerson(img_path=path)
        vec2=face_rec.create_128()
        unknown_vec.append(vec2)
        unknown_names.append(name)
    #2个数据集进行比对

    print('\nWait for the results!  在unknown文件夹里的人脸识别结果将会依次在以下被展示：')
    for i in range(len(unknown_vec)):
        dis=[]
        for j in range(len(vec)):
            #将所有比较值都加进列表，并求其最小值
            compare_value=compare_face(unknown_vec[i],vec[j])
            dis.append(compare_value)
        min_=min(dis)
        min_index=dis.index(min_)
       #输出结果，相当于是k=1时的最邻近法，取欧氏距离最小的样本作为划分的标准；但是k=1有最小近似误差，但是有较大估计误差，所以
        #采取一个阈值0.54，大于0.54的欧氏距离都属于判别失败，属于识别失败，这个精度值依据模型给出；
        #对于精确识别，需要有一个阈值，尽管是最小值但是还需要小于阈值，这样才算做识别成功
        if min_<0.54:
            print('\nMatch success! This man on the pic of {} is {}！'.format(unknown_names[i] + '.jpg', names[min_index]))

        else:
            print('比对失败，可能需要换一张图片！{}'.format(unknown_names[i]))

if __name__=='__main__':
    main()





