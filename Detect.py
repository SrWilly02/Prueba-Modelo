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
model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'model/best.pt', force_reload = True)

# Leer la imagen
image_path = 'captura/captura_chrome3.jpg'
image = cv2.imread(image_path)

# Realizar detecciones
results = model(image)

# Extraer etiquetas y probabilidades
detections = results.pandas().xyxy[0]  # Obtiene el DataFrame con las detecciones
total_detections = len(detections)

# Verificar si hay detecciones
if total_detections > 0:
    phishing_count = detections[detections['name'] == 'Phishing'].shape[0]
    no_phishing_count = detections[detections['name'] == 'No Phishing'].shape[0]

    # Calcular el porcentaje de cada etiqueta
    phishing_percentage = (phishing_count / total_detections) * 100
    no_phishing_percentage = (no_phishing_count / total_detections) * 100

    print(f"Porcentaje de Phishing: {phishing_percentage:.2f}%")
    print(f"Porcentaje de No Phishing: {no_phishing_percentage:.2f}%")
else:
    print("No se detectaron elementos en la imagen.")

# Convertir los resultados en formato de imagen
results_img = np.squeeze(results.render())

# Guardar la imagen con detecciones en formato PNG
output_path = 'resultado/deteccion_resultado.png'
cv2.imwrite(output_path, results_img)

# Mostrar la imagen con detecciones
cv2.imshow('Detecciones YOLOv5', results_img)
cv2.waitKey(0)
cv2.destroyAllWindows
