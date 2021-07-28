
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

# Aplicação web
"""
get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nfrom PIL import Image\nimport numpy as np\n\nimport pickle\n#loading our model\nmodel = pickle.load(open(\'new_model.pkl\',\'rb\'))\n\nPAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}\nst.set_page_config(**PAGE_CONFIG)\n\n\ndef main():\n  st.title("Sistema para identificar casos de Catarata por meio de imagens")\n  st.subheader("Por favor, insira a foto de um olho para diagnostico:")\n  uploaded_file = st.file_uploader("Selecionar imagem")\n  \n  if(uploaded_file != None):\n    st.image(uploaded_file, caption=\'Imagem enviada\')\n    img = Image.open(uploaded_file)\n    img = img.resize((128, 128))\n    new_img = np.asarray(img)\n    new_img = new_img.reshape(1, 49152).astype(\'float32\')\n    new_img /= 255\n    #new_img.shape\n\n    prediction = model.predict(new_img)\n    st.write(prediction)\n\n\n\nif __name__ == \'__main__\':\n  main()')


# !ngrok authtoken 1vrm9JHZZwp8cGvFksYGAFrZYqJ_6YFTSHmdwYAvA6m9CrT2d # Token Yasser

get_ipython().system('ngrok authtoken 1vtlQcqwEjpGfuMGMeSMXAnXzsj_4qQMmbJVcQJTuLHzrUvWt # Token Thiago')


get_ipython().system('ngrok')


from pyngrok import ngrok

get_ipython().system('streamlit run app.py &>/dev/null&')

get_ipython().system('pgrep streamlit')

publ_url = ngrok.connect('8501')

publ_url"""


