# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:36:20 2019
@author: Santiago Lasso
"""
import os
import pandas as pd
import imageio
import numpy as np

path_to_package = 'Datos/'
json_files = [pos_json for pos_json in os.listdir(path_to_package) if pos_json.endswith('.json')]
image_files = [pos_image for pos_image in os.listdir(path_to_package) if pos_image.endswith('.jpg')]
    
"""-------------------------------------------------------------------
*JSON TO DATAFRAME: JSON - PD
-------------------------------------------------------------------"""
def json_to_dataFrame():
    
    list_DataFrame=[]
    
    for i in range(len(json_files)):
    
        datos = pd.read_json(path_to_package+json_files[i])
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
"""----------------------------------------------------------------"""
lists_dataFrames=json_to_dataFrame()
#lists_dataFrames=list_DataFrame
#Une los dataframes; ignore_index=True adecúa los índices entre dataframes
result=pd.concat(lists_dataFrames,ignore_index=True)
"""-------------------------------------------------------------------
*GET VALUES RGB 
-------------------------------------------------------------------"""
def leer_img(image):
    im = imageio.imread(image)
    r,g,b = im[:,:,0],im[:,:,1],im[:,:,2]
    return r,g,b 
"""-------------------------------------------------------------------
*Crear listas de valores RGB de acuerdo a los puntos X y Y
-------------------------------------------------------------------"""
var=0;i=0;pts=0
n_datos=len(result)
    
#Array de valores RGB
mtz_rgb=np.zeros((n_datos,3))
mtz_rgb=mtz_rgb.astype(int)

while i<len(image_files): # i < lista de imagenes (25)
    
    pts_img=len(lists_dataFrames[i]) # Total de puntos por imagen

    if var<n_datos: # Recorre el total de datos (filas)
        
        im = leer_img(path_to_package+image_files[i])
        r = im[0]
        g = im[1]
        b = im[2]
        x = result['cy'][var]
        y = result['cx'][var]

        mtz_rgb[var,0] = (r[x,y])
        mtz_rgb[var,1] = (g[x,y])
        mtz_rgb[var,2] = (b[x,y])
        
        pts+=1    
        
    if pts==pts_img:
        pts=0
        i+=1        
        
    var+=1
"""-------------------------------------------------------------------
#*CONVERT DATAFRAME TO .CVS 
#-------------------------------------------------------------------"""         
#Conversión a dataframe
data_nr=pd.DataFrame(mtz_rgb[:,0],columns=['r'])
data_ng=pd.DataFrame(mtz_rgb[:,1],columns=['g'])
data_nb=pd.DataFrame(mtz_rgb[:,2],columns=['b'])
#Concatena las listas RGB al dataset
result=pd.concat([result,data_nr,data_ng,data_nb],1)
meanDf = pd.DataFrame(result.pop('Zona'))
result = result.join(meanDf)
#Ordena el dataframe por nombre de clase
result=result.sort_values('Zona') 
#Elimina los registro que se etiquetaron mal
result=result.drop(1577,axis=0)   
result=result.drop(26,axis=0)  
#result.to_csv("1.1 dataset.csv", index=False)
