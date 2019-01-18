"""使用百度云接口和人脸库完成本地合影图片的多人脸识别"""
from aip import AipFace
import base64
import dlib
import matplotlib.pyplot as plt
import numpy as np
import math,cv2
import os, glob,math
from skimage import io

"""
人脸特征点检测 栽植模压模压顶替
"""




def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

"""
人脸对齐
"""
def face_alignment(faces):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 用来预测关键点
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
        shape = predictor(np.uint8(face),rec) # 注意输入的必须是uint8类型
       
        order = [36,45,30,48,54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
            #cv2.circle(face, (x, y), 2, (0, 0, 255), -1)    #测试加还是不加

        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x) # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) # 进行放射变换，即旋转
        faces_aligned.append(RotImg)
    return faces_aligned

def feature(path,foces):
    im_raw =cv2.imread(foces).astype('uint8')   
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    src_faces = []
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        detect_face = im_raw[y:y+h,x:x+w]
        src_faces.append(detect_face)
        cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(im_raw, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    faces_aligned = face_alignment(src_faces)
    #cv2.imshow("src", im_raw)    
    for j in os.listdir(path):                 #清空拟装合影照片中分离人脸目录中的文件
        os.remove(path+'\\'+j)         
    i = 0
    for face in faces_aligned:
        #cv2.imshow("det_{}".format(i), face)
        i = i + 1        
        io.imsave(path+'\\'+'Face{}.jpg'.format(i),face)
    #cv2.imshow("Output", im_raw)
    cv2.waitKey(0)

def AipFaceRecognition(pathfile):
               
     with open(pathfile,"rb") as f:  
          # b64encode是编码
          base64_data = base64.b64encode(f.read())
     image = str(base64_data,'utf-8')
     imageType = "BASE64"  #3种"URL"，"FACE_TOKEN"
     groupIdList = "qq"      #你上传百度人脸库照片,用户组ID名叫"qq"
     """ 调用人脸搜索 """
     a=client.search(image, imageType,groupIdList)  
     #print(a['user_list'][2][ 'user_id'],a['user_list'][3][ 'score'])
     return a    #['result']['user_list'][0]['user_id']


if __name__ == "__main__":
     focePath=r'C:\Users\Administrator\Desktop\1234.jpg'   #合影照片
     """合影照片中分离人脸子目录"""
     path = r'zkk' 
     """ 你的 APPID AK SK """
     APP_ID = '15427306'
     API_KEY = 'MUlz7ihrX5BiKcOLo6EGRfbq'
     SECRET_KEY = 'vt0Ob07UWpgOiyiKceacv0IqAzACxsCy'
     client = AipFace(APP_ID, API_KEY, SECRET_KEY)
    
     feature(path,focePath)
     userlist=[]
     for i in os.listdir(path):
          pathfile=path+'\\'+i
          A=AipFaceRecognition(pathfile)
          if A.get('result',None) !=None:
               if A['result']['user_list'][0]['score']>50:    #取相似值大于50%
                    #print('{}照片同{}相似度为{}'.format(i,A['result']['user_list'][0]['user_id'],A['result']['user_list'][0]['score']))
                    print('{}照片同{}相似'.format(i,A['result']['user_list'][0]['user_id']))
 
