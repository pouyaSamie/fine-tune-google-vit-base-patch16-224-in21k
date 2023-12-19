from transformers import ViTImageProcessor, ViTConfig,ViTFeatureExtractor,ViTForImageClassification
import torch.nn.functional as F
import numpy as np
import urllib.parse as parse
import requests
import os
from PIL import Image

sPath_model = './vit-base-beans'

config = ViTConfig.from_pretrained(sPath_model)
model = ViTForImageClassification.from_pretrained(sPath_model, config=config)
processor = ViTImageProcessor.from_pretrained(sPath_model, config=config)
feature_extractor = ViTFeatureExtractor.from_pretrained(sPath_model)

def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)

def predict(image):
  inputs = processor(images=image, return_tensors="pt")
  output = model(**inputs)
  logits = output.logits
  logits_detached = logits.detach().numpy()[0]
  logits_detached_play = logits_detached.copy()

  probabilities = F.softmax(logits, dim=-1)
  pred_class = np.argmax(logits_detached_play)
  probability = probabilities[0, pred_class].item()
  sLabel = config.id2label[pred_class]
 
  return f'with {probability} probability it is {sLabel} '

url = './test.jpeg'
image = load_image(url)
result = predict(image)
print(result)