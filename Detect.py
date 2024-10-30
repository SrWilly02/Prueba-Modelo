# Importar librerías
import torch
import cv2
import numpy as np
import os
import pathlib

# Importante para que Windows sea capaz de emplear PosixPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Cargar el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path= r'C:\Users\USER\Desktop\Prueba Modelo\model\best.pt', force_reload=True) # Para "path", pegar la ruta del modelo

# Leer la imagen
image_path = 'captura/captura_chrome2.png'
image = cv2.imread(image_path)

# Realizar detecciones
results = model(image)

# Convertir los resultados en formato de imagen
results_img = np.squeeze(results.render())

# Mostrar la imagen con detecciones
cv2.imshow('Detecciones YOLOv5', results_img)
cv2.waitKey(0)
cv2.destroyAllWindows
