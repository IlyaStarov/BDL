import ultralytics
from ultralytics import YOLO
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ultralytics.checks()

# Загрузка классификационной модели
model = YOLO('yolov8s.pt') 

# Обучение модели
model.train(data='d:\\lab5_DL\\animals.yaml', model="yolov8s.pt", epochs=1, imgsz=224, batch=16, 
            project='animals_classifier', val = True, verbose=True)


results = model("animals\\val\\images\\pixabay_dog_002239.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model("animals\\val\\images\\pixabay_cat_002662.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model("animals\\val\\images\\pixabay_wild_000838.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model(".\\cat.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model(".\\cat2.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model(".\\wild_and_wild.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model(".\\cat_and_dog.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model(".\\cat_and_dog2.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())


results = model(".\\cat_and_human.jpg")

# посмотрим что получилось
result = results[0]
cv2.imshow("YOLOv8", result.plot())