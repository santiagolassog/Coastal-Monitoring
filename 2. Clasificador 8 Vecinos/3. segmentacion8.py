# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:16:20 2019
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
Cargar, recorrer imagen y obtener valores RGB 
----------------------------------------------------------------------------""" 
#Leer imagen a segmentar
img1 = mg.imread('Resultados/mean.jpg') 
#Limita a leer solo 3 capas de la imagen "0:3"
img1 = img1[:,:,0:3]
#Tamaño de la imagen
fil,col,cap = img1.shape
cont=0; 
#Crear array de valores RGB para predecir
#fil=m*n ; col=27
arr=np.zeros([(fil-2)*(col-2),27])

for i in range(1,(fil-1)):
    for j in range(1,(col-1)):
        #Toma el valor RGB de los 9 pixeles y los aplana en fil=1 col=27
        matriz = img1[i-1:i+2,j-1:j+2,:].reshape(1,27)
        #Asigna el valor de matriz en la posición "cont" del pixel actual
        arr[cont] = matriz 
        cont+=1

"""---------------------------------------------------------------------------- 
Aplicar modelo para predecir
----------------------------------------------------------------------------""" 
#Ruta y nombre del modelo
ruta='2.1 modelo8_RF.pkl' 
#Pasar array de valores RGB para predecir
estado=predecir(arr,ruta)
clasif=estado.reshape(fil-2,col-2)
"""---------------------------------------------------------------------------- 
Segmentación de imagen 
----------------------------------------------------------------------------"""
#Crear matriz de ceros para la imagen segmentada
final=np.zeros((fil-2,col-2,3),'uint8')
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
paleta={'0':(0,0,255),'1':(255,255,0),'2':(255,0,255),'3':(0,255,255),'4':(255,0,0)}
#Recorre la imagen y pinta cada pixel según la clase
for clase in paleta.keys():
    final[clasif == int(clase)] = paleta[clase]
#Hacer plot de la imagen procesada
plt.imshow(final)
#Guardar imagen segmentada
#img_final = Image.fromarray(final)
#img_final.save('Resultados/MeanPROCESS.jpg')
