import cv2
import numpy as np
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

def nothing(x):
    pass


ESC = 27
image_x, image_y = 28, 28

classifier = load_model('models/trained_model_mnist.h5')

def predictor(frame):
            
       test_image = image.img_to_array(frame)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       for x in range(10):
           if result[0][x] == 1:
              return str(x)
      
       

def main() :            

       cam = cv2.VideoCapture(0)
       
       cv2.namedWindow("Trackbars")

       cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
       cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
       cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
       cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
       cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
       cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

       img_text = ''

       while True:
              ret, frame = cam.read()
              frame = cv2.flip(frame,1)
              l_h = cv2.getTrackbarPos("L - H", "Trackbars")
              l_s = cv2.getTrackbarPos("L - S", "Trackbars")
              l_v = cv2.getTrackbarPos("L - V", "Trackbars")
              u_h = cv2.getTrackbarPos("U - H", "Trackbars")
              u_s = cv2.getTrackbarPos("U - S", "Trackbars")
              u_v = cv2.getTrackbarPos("U - V", "Trackbars")
              

              img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

              lower_blue = np.array([l_h, l_s, l_v])
              upper_blue = np.array([u_h, u_s, u_v])
              imcrop = img[102:298, 427:623]
              hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
              mask = cv2.inRange(hsv, lower_blue, upper_blue)
              
              
              cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
              cv2.imshow("TESTE - CNN - MNIST", frame)
              
              mask = cv2.flip(mask, 1)
              cv2.imshow("MASK", mask)
              
                               
              img = cv2.resize(mask, (image_x, image_y))
              img_text = predictor(img)
                     

              if cv2.waitKey(1) == ESC:
                     break
       

       cam.release()
       cv2.destroyAllWindows()


__init__ = main()