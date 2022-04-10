import cv2
import face_recognition
import os
import numpy as np

path="pic2"
images = []
classNames = []
myList = os.listdir(path)
print(myList) # mảng tên file ảnh
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # splitext sẽ tách path ra thành 2 phần, phần trước đuôi mở rộng và phần mở rộng
print(len(images))
print(classNames)

#step encoding
def Mahoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR chuyển sang RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnow = Mahoa(images)
print("Mã hóa thành công")
print(len(encodeListKnow))



