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

#Khỏi động webcam
cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()
    framS = cv2.resize(frame,(0,0),None,fx=0.5,fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    # xác định vị trí khuôn mặt trên cam và encode hình ảnh trên cam
    facecurFrame = face_recognition.face_locations(framS) # Lấy từng khuôn mặt và vị trí hiên tại của nó
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodecurFrame, facecurFrame):  # Lấy từng khuôn mặt và vị trí theo cặp
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)

    cv2.imshow('Nhận diện khuôn mặt', frame)
    if cv2.waitKey(1) == ord("q"):  # độ trễ 1/1000s, nếu bấm q sẽ thoát
        break
cap.release()  # giải phóng camera
cv2.destroyAllWindows()  # thoát tất cả các cửa sổ


