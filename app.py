import streamlit as st
from PIL import Image
import numpy as np

import pickle
#loading our model
model = pickle.load(open('new_model.pkl','rb'))

PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)


def main():
  st.title("Sistema para identificar casos de Catarata por meio de imagens")

  st.title('Sistema meramente acadêmico, NÃO LEVAR EM CONSIDERAÇÃO PARA CASOS REAIS!')
  
  st.subheader("Foto de exemplo")
  
  image = Image.open("./image/example.jpg")
  st.image(image, caption="Foto olho saudável")

  st.subheader("Por favor, insira a foto de um olho para diagnostico:")

  uploaded_file = st.file_uploader("Selecionar imagem")
  
  if(uploaded_file != None):
    st.image(uploaded_file, caption='Imagem enviada')
    img = Image.open(uploaded_file)
    img = img.resize((128, 128))
    new_img = np.asarray(img)
    new_img = new_img.reshape(1, 49152).astype('float32')
    new_img /= 255
    #new_img.shape

    prediction = model.predict(new_img)
    st.write(prediction)



if __name__ == '__main__':
  main()
