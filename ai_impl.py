import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('model.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(32, 32))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return class_names[predicted_class]

img_path = 'th.jpg'
predicted_label = predict_image(img_path)
print(f'The image is predicted to be a: {predicted_label}')

img = tf.keras.utils.load_img(img_path)
plt.imshow(img)
plt.title(f'Predicted: {predicted_label}')
plt.show()
