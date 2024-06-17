print("Importing libs...")
import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

import numpy as np

import keras
import torch
from ultralytics import YOLO

from torchvision import transforms
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

train_dir = os.path.join(".", "dataset", "train")
classes = sorted(os.listdir(train_dir))
print("Imported\n")

print("Loading models...")
model01 = keras.saving.load_model(os.path.join(".", "models", "modelo01.h5"))
precisionByClass01Path = os.path.join(".", "cache", "precisionByClass01.npy")
precisionByClass01 = np.load(precisionByClass01Path)

model02 = torch.jit.load(os.path.join(".", "models", "model02.zip"))
precisionByClass02Path = os.path.join("cache", "precisionByClass02.npy")
precisionByClass02 = np.load(precisionByClass02Path)

yolo = YOLO(os.path.join('.', 'models', 'yolov8n.pt'))
print("Loaded")

def predict01(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_image(img)
  img = tf.image.resize(img, size = [224, 224])
  img = img/255.

  return model01.predict(tf.expand_dims(img, axis=0), verbose=None)

transform02 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(240),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def predict02(image_path):
  image = Image.open(image_path)
  image = transform02(image).unsqueeze(0)
  
  with torch.no_grad():
      model02.eval()
      output = model02(image)
      probabilities = torch.nn.functional.softmax(output[0], dim=0)
  return np.array(probabilities)

def predict04(image_path):
  pred_1 = predict01(image_path) * precisionByClass01
  pred_2 = predict02(image_path) * precisionByClass02
  pred_class = (pred_1 + pred_2).argmax()

  return pred_class, classes[pred_class]

def pred_unprocessed_image(image_path):
  image = cv2.imread(image_path)
  
  model = YOLO(os.path.join('.', 'models', 'yolov8n.pt'))
  results = model.predict(image)
  class_names = model.names

  found = []
  for result in results[0].boxes:
    class_id = int(result.cls)

    if class_names[class_id] == 'bird':
      x1, y1, x2, y2 = map(int, result.xyxy[0])

      cropped_image = image[y1:y2, x1:x2]
      resized_image = cv2.resize(cropped_image, (224, 224))

      output_path = 'crop.jpg'
      cv2.imwrite(output_path, resized_image)

      pred = predict04(output_path)
      found.append(pred)

  return found


test_dir = os.path.join(".", "test")
emojis = ['❌', '✅']

# Shoud be able to identify two different species in the same image
twoDifferentSpeciesImgPath = os.path.join(test_dir, "2especiesDiferentes.jpg")
preds = pred_unprocessed_image(twoDifferentSpeciesImgPath)
preds = [pred[1] for pred in preds]

print("")
print("Shoud be able to identify two different species in the same image: ")
print(f"  - Did find both birds? {emojis[len(preds) == 2]}")
print(f"  - Predicted correctly? {emojis['CRESTED CARACARA' in preds and 'BLACK VULTURE' in preds]}")

# Should be able to identify even if theres text in image
textInImageImgPath = os.path.join(test_dir, "comTexto.jpg")
preds = pred_unprocessed_image(textInImageImgPath)
preds = [pred[1] for pred in preds]


print("")
print("Should be able to identify even if theres text in image?")
print(f"  - Did find the bird? {emojis[len(preds) > 0]}")
print(f"  - Predicted correctly? {emojis['CAATINGA CACHOLOTE' in preds]}")

