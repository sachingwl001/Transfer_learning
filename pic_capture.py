
import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
count = 0
while True:
    status, pic = cap.read()
    facecorr = face.detectMultiScale(pic)
    l = len(facecorr)
    if len(facecorr) == 0:
        pass
    else:
        count = l + count
        x1 = facecorr[0][0] - 50
        y1 = facecorr[0][1] - 50
        x2 = x1 + 224
        y2 = y1 + 224
        c_pic = pic[y1:y2, x1:x2]
        file_path = 'C://Users//asus//Desktop//MLops_WS//dataset//train_set//Rahul//image' + str(count) + '.jpg'
        cv2.imshow('face', c_pic)
        cv2.imwrite(file_path, c_pic)

        if cv2.waitKey(1) == 13:
            break
cv2.destroyAllWindows()
cap.release()