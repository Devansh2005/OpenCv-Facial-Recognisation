
import cv2

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface.xml")

while True:

    ret, frame = cap.read()

    if ret:
        faces = classifier.detectMultiScale(frame)

        for (x,y,w,h) in faces:
            
            win = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("My window", win)

    key = cv2.waitKey(1)

    if key == 13:
        break

cap.release()
cv2.destroyAllWindows()