# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:36:20 2019
@author: usuario
"""
import os
import pandas as pd
import numpy as np
import imageio
import matplotlib.pyplot as plt

path_to_package = 'C:/Users/usuario/Desktop/Clasificación Medias'
json_files = [pos_json for pos_json in os.listdir(path_to_package) if pos_json.endswith('.json')]
image_files = [pos_image for pos_image in os.listdir(path_to_package) if pos_image.endswith('.jpg')]

"""-------------------------------------------------------------------
*JSON TO DATAFRAME: JSON - PD
-------------------------------------------------------------------"""
def json_to_dataFrame():
    
    list_DataFrame=[]
    
    for i in range(len(json_files)):
        
        datos = pd.read_json(json_files[i])
        datos = datos.transpose().reset_index()[['filename','regions']]
        
        list_shape=[];list_zone=[];
        x=datos['regions'][0];y=len(x)
        
        for fil in range (datos.shape[0]):
            for col in range (y):
                data_shape = datos['regions'][fil][col]['shape_attributes']
                data_zone = datos['regions'][fil][col]['region_attributes']
                list_shape.append(data_shape)
                list_zone.append(data_zone)
        dataf1 = pd.DataFrame(list_shape)
        dataf2 = pd.DataFrame(list_zone)
        datos = pd.concat([dataf1,dataf2],1)
        
        #Cambiar posición de las columnas
        cols=datos.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        
        list_DataFrame.append(datos)
    
    return list_DataFrame

lists_dataFrames = json_to_dataFrame()
#Une los dataframes; ignore_index=True adecúa los índices entre dataframes
result = pd.concat(lists_dataFrames,ignore_index=True)
#result = np.array(result)

"""-------------------------------------------------------------------
*GET VALUES RGB 
-------------------------------------------------------------------"""
def leer_img(image):
    im = imageio.imread(image)
    r,g,b = im[:,:,0],im[:,:,1],im[:,:,2]
    return r,g,b 
#-------------------------------------------------------------------
#Crear listas de valores RGB de acuerdo a los puntos X y Y
#-------------------------------------------------------------------
var=0;i=0;pts=0
val_r=[];val_g=[];val_b=[]
n_datos=len(result)
    
while i < len(image_files): # i < lista de imagenes (25)
    
    pts_img=len(lists_dataFrames[i]) # Total de puntos por imagen

    if var<n_datos:
        im = leer_img(image_files[i])
        r = im[0]
        g = im[1]
        b = im[2]
        x = result['cy'][var]
        y = result['cx'][var]
        val_r.append(r[x,y])
        val_g.append(g[x,y])
        val_b.append(b[x,y])
        pts+=1    
#    if pts%pts_img==0:
    if pts==pts_img:
        pts=0
        i+=1        
    var+=1
"""-------------------------------------------------------------------
#*CONVERT DATAFRAME TO .CVS 
#-------------------------------------------------------------------"""         
#Conversión a dataframe
data_nr=pd.DataFrame(val_r,columns=['r'])
data_ng=pd.DataFrame(val_g,columns=['g'])
data_nb=pd.DataFrame(val_b,columns=['b'])
result=pd.concat([result,data_nr,data_ng,data_nb],1)
#Cambiar posición de la Zona
cols=result.columns.tolist()
meanDf = pd.DataFrame(result.pop('Zona'))
result = result.join(meanDf)
#result.to_csv("dataset_media.csv", index=False)
"""-------------------------------------------------------------------
*SEPARACIÓN DE CLASES 
-------------------------------------------------------------------""" 
data = np.array(result)
data_sec=[];data_hum=[];data_rom=[];data_agu=[];data_roc=[]

for i in range(n_datos):
    if data[i,6]=='Seca':
        data_sec.append(data[i,0:])
    if data[i,6]=='Humeda':
        data_hum.append(data[i,0:])
    if data[i,6]=='Rompiente':
        data_rom.append(data[i,0:])
    if data[i,6]=='Agua':
        data_agu.append(data[i,0:])
    if data[i,6]=='Roca':
        data_roc.append(data[i,0:])
##--------------------------------------------------------------
##Listas RGB de cada zona
##--------------------------------------------------------------        
#R_sec=[];R_hum=[];R_rom=[];R_agu=[];R_roc=[]
#G_sec=[];G_hum=[];G_rom=[];G_agu=[];G_roc=[]
#B_sec=[];B_hum=[];B_rom=[];B_agu=[];B_roc=[]
##--------------------------------------------------------------
##Crea una lista RGB para cada zona
##--------------------------------------------------------------
#
##def RGB(r,g,b,list):    
#a=data_roc
#b=a[0]
#for i in range(len(a)): #Recorre total de datos por clase de[0-n]
#    j=3
#    for j in range(len(b)): #Recorre array de datos por fila de[0-6]          
#        if j==3:
#            R_roc.append(data_roc[i][j])
#        elif j==4:
#            G_roc.append(data_roc[i][j])
#        elif j==5: 
#            B_roc.append(data_roc[i][j])
#        j+=1  
#    i+=1
#
#"""-------------------------------------------------------------------
#*HISTOGRAMA DE CADA CLASE 
#-------------------------------------------------------------------"""
##--------------------------------------------------------------
##Genera los histogramas de cada clase según los valores RGB
##--------------------------------------------------------------
#def lbl_hist(clase,n_channel):
#    #    n,bins,patches=plt.hist(datos,num_bins,facecolor=color)
#    #    plt.show()
#    plt.title('Histograma ' + clase + ' - Canal: '+ n_channel)
#    plt.xlabel('Valores de intensidad')
#    plt.ylabel('Numero de pixeles')
#
#plt.hist(R_roc,bins=10,facecolor='red',alpha=0.7)
##ruta='C:/Users/usuario/Desktop/Clasificación Medias/Histogramas/1.hist_Rroc.png'
##lbl_hist('Roca','R')
##plt.savefig(ruta,format='png')
#
#plt.hist(G_roc,bins=10,facecolor='green',alpha=0.7)
##ruta='C:/Users/usuario/Desktop/Clasificación Medias/Histogramas/2.hist_Groc.png'
##lbl_hist('Roca','G')
##plt.savefig(ruta,format='png')
#
#plt.hist(B_roc,bins=10,facecolor='blue',alpha=0.7)
##ruta='C:/Users/usuario/Desktop/Clasificación Medias/Histogramas/3.hist_Broc.png'
##lbl_hist('Roca','B')
##plt.savefig(ruta,format='png')
#
#ruta='C:/Users/usuario/Desktop/Clasificación Medias/Histogramas/4.hist_RGBroc.png'
#plt.savefig(ruta,format='png')
#plt.title('Histograma RGB - Clase Roca')
#plt.xlabel('Valores de intensidad')
#plt.ylabel('Numero de pixeles')
#plt.savefig(ruta,format='png')
