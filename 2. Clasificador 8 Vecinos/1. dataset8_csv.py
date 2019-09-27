# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:16:20 2019
@author: Santiago Lasso
"""
#librerías
import os
import pandas as pd
import imageio
import numpy as np
#Directorio datos: archivos json e imágenes
path = 'Datos/'
#Recorre el directorio de archivos y separa en listas independientes los archivos
json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
image_files = [pos_image for pos_image in os.listdir(path) if pos_image.endswith('.jpg')]
"""--------------------------------------------------------------------------------------------------
*JSON TO DATAFRAME: JSON - PD
--------------------------------------------------------------------------------------------------"""
def json_to_dataFrame(): 
    
    list_DataFrame=[]   
    
    for i in range(len(json_files)):       
        #Leer leer .json según posición en la lista
        datos = pd.read_json(path+json_files[i])
        datos = datos.transpose().reset_index()[['filename','regions']]
        #Listas de regiones del .json
        list_shape=[];list_zone=[];
        x=datos['regions'][0];y=len(x)   
        
        #Ciclo que extrae los datos del json y genera un dataframe
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
        #Agrega el dataframe de cada json a una lista
        list_DataFrame.append(datos)
        
    return list_DataFrame
"""--------------------------------------------------------------------------------------------------"""
lists_dataFrames = json_to_dataFrame()
#Une los dataframes; ignore_index=True ordena los índices entre dataframes
result = pd.concat(lists_dataFrames,ignore_index=True)
"""--------------------------------------------------------------------------------------------------
*GET VALUES RGB 
--------------------------------------------------------------------------------------------------""" 
#Variables para recorrer las listas de archivos
var=0;i=0;pts=0
n_datos=len(result)
#Array de ceros para valores RGB
mtz_rgb=np.zeros([n_datos,27])

while i<len(image_files): # i<lista de imagenes (25)       
    pts_img=len(lists_dataFrames[i]) # Total de puntos por imagen    
    if var<n_datos:       
        #Lee la imagen actual
        im = imageio.imread(path+image_files[i])
        #Toma los valores X y Y del punto pixel principal
        x = result['cy'][var]
        y = result['cx'][var]
        #Toma el valor RGB de los 9 pixeles y los aplana en fil=1 col=27
        matriz = im[x-1:x+2,y-1:y+2,:].reshape(1,27)
        #Asigna los valores RGB de los 9 pixels a fila[var] de la matriz vacía
        mtz_rgb[var] = matriz               
        pts+=1          
    #Condicional del contador de puntos por imagen    
    if pts==pts_img: 
        pts=0
        i+=1           
    var+=1
"""--------------------------------------------------------------------------------------------------
*CONVERT DATAFRAME TO .CVS 
--------------------------------------------------------------------------------------------------"""
#Convertir array RGB a dataframe
mtz_rgb = pd.DataFrame(mtz_rgb)
#Concatenar dataframe RGB con dataframe principal
result=pd.concat([result,mtz_rgb],1)
#Cambiar posición de la Zona
meanDf = pd.DataFrame(result.pop('Zona'))
result = result.join(meanDf)
#Ordenar el dataframe por cada clase
result=result.sort_values('Zona') 
#Eliminar el registros erróneos
result=result.drop(1577,axis=0)   
result=result.drop(26,axis=0)  
#Exportar y guardar .csv
#result.to_csv("1.1 dataset8.csv", index=False)