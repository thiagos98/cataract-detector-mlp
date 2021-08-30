
from IPython import get_ipython

from matplotlib import pyplot
from matplotlib.image import imread
import numpy as np
from PIL import Image
import operator
from operator import itemgetter
import os
from sklearn.neural_network import MLPClassifier


# folder = '/content/drive/My Drive/Colab Notebooks/ia/treino_olhos/normal/'
folder = './treino_olhos/normal/'

for i in range(6):
    pyplot.subplot(330 + 1 + i)
    filename = 'image'+str(i)+'.jpg'
    image = imread(folder+"/"+filename)
    pyplot.imshow(image)
pyplot.show()


# folder = '/content/drive/My Drive/Colab Notebooks/ia/treino_olhos/cataract/'
folder = './treino_olhos/cataract/'

for i in range(6):
    pyplot.subplot(330 + 1 + i)
    filename = 'image'+str(i)+'.jpg'
    image = imread(folder+"/"+filename)
    pyplot.imshow(image)
pyplot.show()


x = []
y = []

count = 0
# dir = '/content/drive/My Drive/Colab Notebooks/ia/treino_olhos/'
dir = './treino_olhos/'

for i in os.listdir(dir):
  path = dir+'/'+i
  print(i, ':', len(os.listdir(path)))
  count += len(os.listdir(path))
  for j in os.listdir(path):
    img = Image.open(path + '/' + j)
    img = img.resize((128, 128))
    x.append(np.asarray(img))
    y.append(i)

print(count)
x = np.asarray(x)
y = np.asarray(y)
print(x.shape, y.shape)


x = x.reshape(x.shape[0], 49152).astype('float32')
x /= 255

mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=500)
mlp.fit(x,y)

# img = Image.open('/content/drive/My Drive/Colab Notebooks/ia/teste/normal/image75.jpg')
img = Image.open('./teste/normal/image75.jpg')
img = img.resize((128, 128))

new_img = np.asarray(img)
new_img = new_img.reshape(1, 49152).astype('float32')

new_img /= 255
new_img.shape

prediction = mlp.predict(new_img)
print(f'Olho Saudável: {prediction}')

print(f'Score: {mlp.score(x, y)}')


# img = Image.open('/content/drive/My Drive/Colab Notebooks/ia/teste/cataract/image0.jpg')
img = Image.open('./teste/cataract/image0.jpg')
img = img.resize((128, 128))


new_img = np.asarray(img)
new_img = new_img.reshape(1, 49152).astype('float32')

new_img /= 255
new_img.shape

prediction = mlp.predict(new_img)
print(f'Olho Com Catarata: {prediction}')

import pickle
pickle.dump(mlp,open('new_model.pkl','wb')) #saving our model in .pkl file
