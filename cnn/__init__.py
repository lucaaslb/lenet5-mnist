"""
author: Lucas Lacerda @lucaaslb

Contém as implementações de arquiteturas CNN.
 
[LeNet5] - CNN inspirada na arquitetura de LeCun

Referencias: 

[1] http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf # Modelo original da LeNet5
[2] https://keras.io/layers/convolutional/              # Documentação do Keras
[3] https://keras.io/visualization/  
"""
 
# importar os pacotes necessários
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
 
class LeNet5(object):
    """
    Arquitetura LeNet5
 
    Com foco no reconhecimento de dígitos (MNIST), esta CNN é composta
    por uma sequência contendo os seguintes layers:
 
    INPUT => CONV => POOL => CONV => POOL => FC => FC => OUTPUT
    """
    @staticmethod
    def build(width, height, channels, classes):
        """
        Constroi uma CNN com arquitetura LeNet5.
 
        :param width: Largura em pixel da imagem.
        :param height: Altura em pixel da imagem.
        :param channels: Quantidade de canais da imagem.
        :param classes: Quantidade de classes para o output.
        :return: Cnn do tipo LeNet5.
        
        """
        inputShape = (height, width, channels) #formato da imagem
 
        model = Sequential()
        model.add(Conv2D(filters = 6, kernel_size = (5, 5), padding = "same",
                         input_shape = inputShape)) #Camada de convolucao: Com 6 filtros e Kernel 5x5,  padding = "same" resulta em preenchimento da entrada de forma que a saída tenha o mesmo comprimento que a entrada original. 
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(16, (5, 5)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation("relu"))
        model.add(Dense(84))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
 
        return model
