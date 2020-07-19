import cv2
# import numpy as np

#Cascade Classifier frontal face xml file from github 

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface.xml")

def face_extractor(img):

    # convert rgb to gray
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces= face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is None:  # if faces is(): ---> If there is no face found
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

        return cropped_face


cap= cv2.VideoCapture(0)

#click number of photos
count = 0

while True:
    ret, frame= cap.read()
    if face_extractor(frame) is not None :
        count +=1
        face = cv2.resize(face_extractor(frame), (300,300))
        
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        file_name_path = "C:/Users/Devansh/Desktop/Facial Recognisation/My face/sample" +str(count)+".jpg" # path to store sample pics 

        cv2.imwrite(file_name_path, face)  # to write in folder

        cv2.putText(face, str(count),(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255), 2) #counting on image
        cv2.imshow("Face Cropper", face)
    else:
        print("Face Not Found")

        pass
    if cv2.waitKey(1) ==13 or count == 500:  #press enter or It'll collect 100 samples
        break

cap.release()
cv2.destroyAllWindows()
print("\nCollecting Samples Complete !!!\n")




