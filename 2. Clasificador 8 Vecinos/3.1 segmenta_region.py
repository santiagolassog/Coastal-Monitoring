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
Cargar imagen: Definir región a procesar y obtener sus valores RGB 
----------------------------------------------------------------------------""" 
#Leer imagen a segmentar
img1 = mg.imread('Resultados/mean_mc.jpg') 
#Limita a leer solo 3 capas de la imagen (0:3)
img1 = img1[:,:,0:3]
#Tamaño de la imagen
fil,col,cap = img1.shape
#Dimensiones de la región a procesar
#filx=154 ; colx=378
filx=147 ; colx=718

#Coordenadas (fil*col) de la región a procesar
#xmin=181 ; xmax=335
#ymin=179 ; ymax=557
xmin=188 ; xmax=335
ymin=1 ; ymax=719

#Crear array de valores RGB para predecir
arr=np.zeros([(filx)*(colx),27])
cont=0;
for i in range(xmin,xmax):
    for j in range(ymin,ymax):
        #Toma el valor RGB de los 9 pixeles y los aplana en fil=1 col=27
        matriz = img1[i-1:i+2,j-1:j+2,:].reshape(1,27)
        #Asigna el valor de matriz en la posición "cont" del pixel actual
        arr[cont] = matriz 
        cont+=1
"""---------------------------------------------------------------------------- 
Aplicar modelo para predecir
----------------------------------------------------------------------------""" 
#Ruta y nombre del modelo
ruta='2.1 modelo8_7135.pkl' 
#Pasar array de valores RGB para predecir
estado=predecir(arr,ruta)
clasif=estado.reshape(filx,colx)
"""---------------------------------------------------------------------------- 
Segmentación de imagen 
----------------------------------------------------------------------------"""
#Crear matriz de ceros para la imagen segmentada
final=np.zeros((fil,col,3),'uint8')
#Asigna imagen original al array final
final=img1
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

#Guarda en la banda 0 del array(final) los valores de clasif(predicción)
final[xmin:xmax,ymin:ymax,0]=clasif

#Pinta los pixels de la región de prueba según valor de predicción
for clase in paleta.keys():
    final[final[:,:,0]==int(clase)] = paleta[clase]

#Hacer plot de la imagen procesada
plt.imshow(final)
    
#Guardar imagen segmentada
img_final = Image.fromarray(final)
img_final.save('Resultados/mean_mc2.jpg')

"""---------------------------------------
Valores individuales para pintar por clase
------------------------------------------
#final[final[:,:,0]==0] = (0,0,255)
#final[final[:,:,0]==1] = (255,255,0)
#final[final[:,:,0]==2] = (255,0,255)
#final[final[:,:,0]==3] = (0,255,255)
#final[final[:,:,0]==4] = (255,0,0)
---------------------------------------"""
