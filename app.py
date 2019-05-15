"""
author: Lucas Lacerda @lucaaslb

Exemplo para preditar uma imagem com um numero a partir do modelo treinado, exemplo de imagens no diretorio './images'

Executar:

python3 app.py 'local_imagem'

*deve ser informado o path absoluto da imagem

""" 

import cv2
import numpy as np
import sys 
from keras.models import load_model
from keras.preprocessing import image

image_x, image_y = 28, 28

classifier = load_model('models/trained_model_mnist.h5')

def predictor(img):
            
       test_image = image.img_to_array(img)       
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       maior, class_index = -1, -1

       for x in range(10):      
           
           if result[0][x] > maior:
              maior = result[0][x]
              class_index = x
       
       return [result, class_index]

def main() :    
       
       path_img = str(sys.argv[1])              
       img = cv2.imread(path_img)       
       
       img = cv2.resize(img, (image_x, image_y))       
       imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY, dstCn=1)
   
       predict = predictor(imggray)
       
       print('\n\n===========================\n')
       print('Imagem: ', path_img)
       print('Vetor de resultado: ', predict[0])
       print('Classe: ', predict[1])       
       print('\n===========================\n')
      

__init__ = main()