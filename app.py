import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import platform
import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

#title
st.title("Transportni klassifikatsiya qiluvchi model")
st.text('Be carefull! You should load three kind of photos which are Car, Airplane and Boat on it!')

#rasm joylash
file = st.file_uploader('Rasm yuklash', type=['png','jpg','gif','svg'])
if file:
  st.image(file)

  #PIL Convert
  img = PILImage.create(file)

  #model
  model = load_learner("transport_model.pkl")

  #prediction
  pred, pred_id, probs = model.predict(img)
  st.success(f"Bashorat: {pred}")
  st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

  #plotting
  fig = px.bar(x=probs*100, y=model.dls.vocab)
  st.plotly_chart(fig)
  
