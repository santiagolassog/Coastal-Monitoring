# Coastal-Monitoring
Procesamiento de imágenes costeras 

El proyecto consiste en procesar imágenes de la costa, con el fin de segmentar las distintas zonas de la playa, tales como:
-1)Arena Seca 
-(2)Arena Húmeda 
-3)Rompiente 
-4)Agua 
-5)Roca

La fuente de datos pertenecen al Instituro Argentino de Oceanografía (IADO). Hay algunas cámaras ubicadas en el balneario de Pehuen-Có-Buenos Aires(Argentina). Los datos manejados en este proyecto corresponden a videos e imágenes de la cámara frontal ubicada en este lugar.

Etiquetado:
Con ayuda de la plataforma VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/via.html), se realizó el etiquetado manual de los datos a procesar y entrenar el clasificador.

Dataset y clasificador: como set de datos se tomó de referencia 25 imágenes, y un total de 6832 puntos.

En el repositorio se encuentran disponibles 2 clasificadores, junto con los datos entrenados. Un clasificador de datos principal (pixel a pixel), y otro más robusto tomando referencia los 8 pixeles vecinos al principal.
