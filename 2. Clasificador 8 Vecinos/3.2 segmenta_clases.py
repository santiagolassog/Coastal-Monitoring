# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:05:59 2019
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
ruta='2.1 modelo8_100.pkl' 
#Pasar array de valores RGB para predecir
estado=predecir(arr,ruta)
clasif=estado.reshape(filx,colx)
"""---------------------------------------------------------------------------- 
Segmentación de imagen 
----------------------------------------------------------------------------"""
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
#paleta={'0':(0,0,255),'1':(255,255,0),'2':(255,0,255),'3':(0,255,255),'4':(255,0,0)}

list_color=[(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,0,0)]

def pinta_clase(clase,name):
    #Crear matriz de ceros para la imagen segmentada
    final=np.zeros((fil,col,3),'uint8')
    #Asigna imagen original al array final
    final=mg.imread('Resultados/mean_mc.jpg') 
    final = final[:,:,0:3]
    #Guarda en la banda 0 del array(final) los valores de clasif(predicción)
    final[xmin:xmax,ymin:ymax,0]=clasif    
    #Reabrir imagen original para asignar valores diferentes a la clase evaluada
    img1 = mg.imread('Resultados/mean_mc.jpg') 
    img1 = img1[:,:,0:3]

    #Asigna valores RGB a los pixeles fuera de la región de interés
    for i in range(xmin,xmax):
        for j in range(ymin,ymax):
            if final[i,j,0]!=clase:
                final[i,j,:]=img1[i,j,:]
                
    #Asignar color a los pixeles según clase evaluada
    final[final[:,:,0]==clase]=list_color[clase]
    #Guardar imagen segmentada
    img_final = Image.fromarray(final)
    plt.imshow(img_final)
    img_final.save('Resultados/'+name)              

pinta_clase(0,'mean_mcAGUA.jpg')
pinta_clase(1,'mean_mcHUMEDA.jpg')
pinta_clase(2,'mean_mcROCA.jpg')
pinta_clase(3,'mean_mcROMPIENTE.jpg')
pinta_clase(4,'mean_mcSECA.jpg')

"""---------------------------------------
Valores individuales para pintar por clase
------------------------------------------
#final[final[:,:,0]==0] = (0,0,255)
#final[final[:,:,0]==1] = (255,255,0)
#final[final[:,:,0]==2] = (255,0,255)
#final[final[:,:,0]==3] = (0,255,255)
#final[final[:,:,0]==4] = (255,0,0)
---------------------------------------"""
