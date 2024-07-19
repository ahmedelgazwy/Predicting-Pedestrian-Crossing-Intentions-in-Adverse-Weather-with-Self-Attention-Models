import tarfile
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import requests
import wget

#path = './Ahmed_Elgazwy/transformers/vit_b8_classification_1.tar.gz'
#path_to_downloaded_file = tf.keras.utils.get_file("saved_model",path, archive_format='tar', untar=True)


#model = tf.keras.models.load_model(path_to_downloaded_file)
#print(model.summary())

def preprocess_image(image):
    image = np.array(image)
    image_resized = tf.image.resize(image, (224, 224))
    image_resized = tf.cast(image_resized, tf.float32)
    image_resized = (image_resized - 127.5) / 127.5
    return tf.expand_dims(image_resized, 0).numpy()

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = preprocess_image(image)
    return image
url='https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt'
filename = wget.download(url)

with open(filename, "r") as f:
    lines = f.readlines()
imagenet_int_to_str = [line.rstrip() for line in lines]
img_url1='https://cdn.britannica.com/60/8160-050-08CCEABC/German-shepherd.jpg'
img_url = "https://p0.pikrepo.com/preview/853/907/close-up-photo-of-gray-elephant.jpg"
image = load_image_from_url(img_url1)

plt.imshow((image[0] + 1) / 2)
plt.show()
model_url = "https://tfhub.dev/sayakpaul/vit_s16_classification/1"

classification_model = tf.keras.Sequential(
    [hub.KerasLayer('./Ahmed_Elgazwy/transformers/vit_b8_classification_1')]
)  
predictions = classification_model.predict(image)
predicted_label = imagenet_int_to_str[int(np.argmax(predictions))]
print(predicted_label)

