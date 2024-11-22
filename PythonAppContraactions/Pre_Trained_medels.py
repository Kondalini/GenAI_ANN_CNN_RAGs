
from keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

# Load the Vgg16 pre-trained models and specify the input shape
model = VGG16(weights='imagenet', include_top=True,input_shape=(224,224,3))

# Loading an image and preprocess it for the model

img_path = '/content/perf.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

# Use the model to predict the class
preds = model.predict(x)

preds
# Get the top 5 predictions with class names
decode_preds = decode_predictions(preds, top=10)[0]
for pred in decode_preds:
  print(pred[1], ":", pred[2])