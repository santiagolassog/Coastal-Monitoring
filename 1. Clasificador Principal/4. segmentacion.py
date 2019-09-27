# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:36:20 2019
@author: Santiago Lasso
"""
#librerías
import numpy as np
import joblib
import imageio as mg
import matplotlib.pyplot as plt
from PIL import Image
"""---------------------------------------------------------------------------- 
Función para cargar modelo
----------------------------------------------------------------------------"""
svm_model = None
def cargar_modelo(ruta):
     svm_model = joblib.load(ruta)  
     return svm_model
"""---------------------------------------------------------------------------- 
Función para predecir
----------------------------------------------------------------------------""" 
def predecir(X,ruta):
    global svm_model
    if svm_model is None:
        svm_model = cargar_modelo(ruta)
    return svm_model.predict(X)
"""---------------------------------------------------------------------------- 
Aplicar modelo para predecir
----------------------------------------------------------------------------""" 
#Ruta de la imagen a segmentar
img1 = mg.imread('Resultados/0108_0152_18_09_04_16_30_mean_mc.jpg')
shape=img1.shape
x1=img1.reshape(-1,3)
#Ruta del modelo
ruta='3.1 modelo.pkl'
estado=predecir(x1,ruta)
clasif=estado.reshape(shape[0],shape[1])
"""---------------------------------------------------------------------------- 
Segmentación de imagen 
----------------------------------------------------------------------------"""
#Crear matriz de ceros para la imagen segmentada
final=np.zeros((shape[0],shape[1],3),'uint8')
"""-----------------------------------
DESCRIPCIÓN DE COLORES:
 ------------------------------------- 
#1: Agua      = (0,0,255)   #Azul  
#2: Húmeda    = (255,255,0) #Amarillo
#3  Roca      = (255,0,255) #Magenta
#4  Rompiente = (0,255,255) #Cyan
#5  Seca      = (255,0,0)   #Rojo
-----------------------------------"""
#Diccionario de colores según clase
paleta = {'0':(0,0,255), '1':(255,255,0), '2':(255,0,255), '3':(0,255,255), '4': (255,0,0)}
#Recorre la imagen y pinta cada pixel según la clase
for clase in paleta.keys():
    final[clasif == int(clase)] = paleta[clase]
#Hacer plot de la imagen procesada
plt.imshow(final)
#Guardar imagen segmentada
img_final = Image.fromarray(final)
img_final.save('Resultados/0108_0152_18_09_04_16_30_mean_mcPROCESS.jpg')
