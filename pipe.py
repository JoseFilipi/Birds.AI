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

p, axs = plt.subplots(2, 2, figsize=(10, 5))
axs[0][0].axis('off')
axs[0][1].axis('off')
axs[1][0].axis('off')
axs[1][1].axis('off')

def pipe():
  pipe_dir = os.path.join(".", "pipes")

  to_valid_dir = os.path.join(pipe_dir, "human", "to-validate")
  unknown_dir = os.path.join(pipe_dir, "human", "unknown")
  to_train_dir = os.path.join(pipe_dir, "to-train")

  images_to_valid = os.listdir(to_valid_dir)

  for filename in images_to_valid:
    image_path = os.path.join(to_valid_dir, filename)
    image = cv2.imread(image_path)
  
    results = yolo.predict(image)
    class_names = yolo.names

    axs[0][0].set_title('Imagem original', fontsize=12)
    axs[0][0].imshow(mpimg.imread(image_path))

    for result in results[0].boxes:
        class_id = int(result.cls)
        if class_names[class_id] == 'bird':
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            cropped_image = image[y1:y2, x1:x2]
            resized_image = cv2.resize(cropped_image, (224, 224))

            output_path = 'crop.jpg'
            cv2.imwrite(output_path, resized_image)

            pred = predict04(output_path)

            axs[0][1].set_title('Pássaro à prever', fontsize=12)
            axs[0][1].imshow(mpimg.imread("crop.jpg"))

            pred_example = mpimg.imread(os.path.join(".", train_dir, pred[1], "001.jpg"))
            axs[1][1].axis('off')
            axs[1][1].set_title('Provável espécie: ' + pred[1], fontsize=12)
            axs[1][1].imshow(pred_example)
            p.show()

            res = input("A classe está correta? [Y/N] ")
            if res.lower() == 'y':
              class_dir = os.path.join(to_train_dir, pred[1])
              os.makedirs(class_dir, exist_ok=True)
              os.rename('crop.jpg', os.path.join(class_dir, f'{len(os.listdir(class_dir))+1:03}.jpg'))
            else:
              res = input("Você sabe a espécie correta? [Y/N] ")
              if res.lower() == "y":
                res = input("Digite o nome da espécie correta: ")
                while res not in classes:
                   res = input(f"{res} não é uma espécie conhecida. Por favor, digite uma espécie válida: ")
                
                os.makedirs(os.path.join(train_dir, res), exist_ok=True)
                os.rename('crop.jpg', os.path.join(train_dir, res, f'{len(os.listdir(res))+1:03}.jpg'))
              else:
                os.rename('crop.jpg', os.path.join(unknown_dir, f'{len(os.listdir(unknown_dir))+1:03}.jpg'))

    already_validate_path = os.path.join(pipe_dir, "human", "already-validated", filename)
    os.rename(image_path, already_validate_path)

pipe()
