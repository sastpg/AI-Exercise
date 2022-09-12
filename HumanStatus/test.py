
# coding: utf-8

# In[5]:


import pandas as pd
from openvino.inference_engine import IECore
import cv2 as cv
import numpy as np
import math


# In[4]:




def sleepiness_detection_demo():
    #人脸检测预处理模型加载
    #初始化推理引擎
    ie = IECore()

    model_xml = "face-detection-0202/face-detection-0202.xml"
    model_bin = "face-detection-0202/face-detection-0202.bin"

    #加载IR文件
    net = ie.read_network(model=model_xml, weights=model_bin)
    #配置输入输出
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape

    #加载可执行网络
    exec_net = ie.load_network(network=net, device_name="CPU")

    #人脸检测点预处理模型加载
    lm_xml = "facial-landmarks-35-adas-0002/facial-landmarks-35-adas-0002.xml"
    lm_bin = "facial-landmarks-35-adas-0002/facial-landmarks-35-adas-0002.bin"

    lm_net = ie.read_network(model=lm_xml, weights=lm_bin)
    lm_input_blob = next(iter(lm_net.input_info))
    lm_output_blob = next(iter(lm_net.outputs))

    ln, lc, lh, lw = lm_net.input_info[lm_input_blob].input_data.shape

    lm_exec_net = ie.load_network(network=lm_net, device_name="CPU")

    #人眼检测（利用opencv自带检测器）
    #创建一个级联分类器对象，加载xml检测器
    eye_xml = cv.CascadeClassifier('D:/software/openvino2021/openvino_2021.2.185/opencv/etc/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    

    #捕捉摄像头的帧
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    while True:
        #读取视频每一帧
        ret, frame = cap.read()
        if ret is not True:
            break
        image = cv.resize(frame, (w, h))
        # 转置 若升维度（HWC->NCHW）：np.expand_dims(img, 0)
        image = image.transpose(2, 0, 1)
        #推理 将image变为四维
        res = exec_net.infer(inputs={input_blob: [image]})
        #获得推理计算后的输出
        res = res[out_blob]

        ih, iw, ic = frame.shape
        for obj in res[0][0]:
            if obj[2] > 0.25:
                #将浮点数转换为实际宽高
                xmin = int(obj[3] * iw)
                ymin = int(obj[4] * ih)
                xmax = int(obj[5] * iw)
                ymax = int(obj[6] * ih)
                #防止越界
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax >= iw:
                    xmax = iw - 1
                if ymax >= ih:
                    ymax = ih - 1
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                #提取人脸部分，获取人脸检测点
                roi = frame[ymin:ymax, xmin:xmax, :]
                rh, rw, rc = roi.shape
                roi_img = cv.resize(roi, (lw, lh))
                roi_img = roi_img.transpose(2, 0, 1)
                lm_res = lm_exec_net.infer(inputs={lm_input_blob: [roi_img]})
                prob_landmarks = lm_res[lm_output_blob]
                #创建空list，存储嘴部检测点的坐标值
                marklist = []
                for index in range(0, len(prob_landmarks[0]), 2):
                    x = np.int(prob_landmarks[0][index] * rw)
                    y = np.int(prob_landmarks[0][index+1] * rh)

                    if (index == 16 or index == 18 or index == 20 or index == 22):
                        marklist.append(x)
                        marklist.append(y)

                #计算嘴部纵横比
                dist_w = math.sqrt(math.pow(marklist[2]-marklist[0], 2) + math.pow(marklist[3]-marklist[1], 2))
                dist_h = math.sqrt(math.pow(marklist[6]-marklist[4], 2) + math.pow(marklist[7]-marklist[5], 2))
                rate = dist_h / dist_w
                #灰度处理
                face_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                #眼睛识别
                eyes = eye_xml.detectMultiScale(face_gray)
                #判断疲劳并作出输出
                if ( rate > 0.4):#len(eyes) < 2 or
                    cv.putText(frame, "tired", (xmin, ymin), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv.putText(frame, "normal", (xmin, ymin), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("sleepiness_detection_demo", frame)
        #停留时读取键值，按Esc键退出
        c = cv.waitKey(1)
        if c == 27:
            break
    cap.release()
    
if __name__ == "__main__":
    sleepiness_detection_demo()


print("done")