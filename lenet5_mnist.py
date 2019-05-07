"""
Treina uma CNN com o dataset MNIST.
 
A CNN é inspirada na arquitetura LeNet-5
""" 
# importar pacotes necessários
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import backend
from keras.datasets import mnist
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import h5py
import time
from cnn import LeNet5     #classe criada na pasta cnn

# importar e normalizar o dataset MNIST
# input image dimensions
imgX, imgY = 28, 28
EPOCHS = 20

print('[INFO] Download dataset: MNIST')

# dividir o dataset entre train (60000) e test (10000)
(trainX, trainY), (testX, testY) =  mnist.load_data() 


print('[INFO] Padronizando imagens de acordo com a lib utilizada em backend pelo Keras')
if backend.image_data_format() == "channels_last": #Tensorflow backend
    print('[INFO] Tensorflow')
    trainX = trainX.reshape(trainX.shape[0], imgX, imgY, 1)
    testX = testX.reshape(testX.shape[0], imgX, imgY, 1)
    input_shape = (imgX, imgY, 1)
else: #Theano backend
    print('[INFO] Theano')
    trainX = trainX.reshape(trainX.shape[0], 1, imgX, imgY)
    testX = testX.reshape(testX.shape[0], 1, imgX, imgY)
    input_shape = (1, imgX, imgY)

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255
testX /= 255

# Transformar labels em vetores binarios
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# inicializar e otimizar modelo
print("[INFO] inicializando e otimizando a CNN...")
begin = time.time()

model = LeNet5.build(28, 28, 1, 10)
model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
              metrics=["accuracy"])
 
# treinar a CNN
print("[INFO] treinando a CNN...")
H = model.fit(trainX, trainY, batch_size=128, epochs=EPOCHS, verbose=1,
          validation_data=(testX, testY))

print("[INFO] Salvando modelo treinado ...")
model.save('models/trained_model_mnist.h5')

end = time.time()

print("[INFO] tempo de execução da CNN: %.2f s" %(end - begin))

# avaliar a CNN
print("[INFO] avaliando a CNN...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(label) for label in range(10)]))
score = model.evaluate(testX, testY, verbose=1)
print('[INFO] Test Loss: ', score[0])
print('[INFO] Test Accuracy: ', score[1])

print("[INFO] Plot loss e accuracy para os datasets 'train' e 'test'")
# plotar loss e accuracy para os datasets 'train' e 'test'
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('models/cnn_mnist.png', bbox_inches='tight')