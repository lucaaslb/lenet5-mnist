import cv2
import numpy as np

def nothing(x):
    pass

image_x, image_y = 28, 28

from keras.models import load_model
classifier = load_model('models/trained_model.h5')

def predictor(frame):
       import numpy as np
       from keras.preprocessing import image
       
      # test_image = image.load_img('temp/verify.png', target_size=(image_x, image_y)).convert('L')
       test_image = image.img_to_array(frame)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       print(result)
       if result[0][0] == 1:
              return '0'
       elif result[0][1] == 1:
              return '1'
       elif result[0][2] == 1:
              return '2'
       elif result[0][3] == 1:
              return '3'
       elif result[0][4] == 1:
              return '4'
       elif result[0][5] == 1:
              return '5'
       elif result[0][6] == 1:
              return '6'
       elif result[0][7] == 1:
              return '7'
       elif result[0][8] == 1:
              return '8'
       elif result[0][9] == 1:
              return '9'
      
       

def main() :            

       cam = cv2.VideoCapture(0)

       img_text = ''
       
       while True:
              ret, frame = cam.read()
              frame = cv2.flip(frame,1)

              img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

              #     lower_blue = np.array([l_h, l_s, l_v])
              #     upper_blue = np.array([u_h, u_s, u_v])
              lower_blue = np.array([103, 38, 60])
              upper_blue = np.array([179, 255, 255])
              imcrop = img[102:298, 427:623]
              hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
              mask = cv2.inRange(hsv, lower_blue, upper_blue)
              
              cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 3.5, (0, 255, 127))
              cv2.imshow("test", frame)
              
              mask = cv2.flip(mask, 1)
              cv2.imshow("mask", mask)
              
                     
              img_name = "verify.png"
              save_img = cv2.resize(mask, (image_x, image_y))
              
              #cv2.imwrite("temp/" + img_name, save_img)    
              img_text = predictor(save_img)
                     

              if cv2.waitKey(1) == 27:
                     break


       cam.release()
       cv2.destroyAllWindows()


__init__ = main()