import cv2 as cv

capture = cv.VideoCapture(0) #Pour ouvrir camera

#accessing pretrained model
pretrained_model = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml") 

while True:
    boolean, frame = capture.read()
    if boolean == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        coordinate_list = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) 
        
        # drawing rectangle in frame
        for (x,y,w,h) in coordinate_list:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
        # Display detected face
        cv.imshow("Live Face Detection", frame)
        
        # condition to break out of while loop
        if cv.waitKey(20) == ord('x'):
            break
        
capture.release()
cv.destroyAllWindows()