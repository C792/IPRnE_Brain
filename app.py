from flask import Flask, request, render_template
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return vgg16process(filename)

def vgg16process(filename:str):
    IMG_SIZE = 128
    import tensorflow as tf
    model = tf.keras.models.load_model('model.h5')

    from tensorflow.keras.preprocessing import image
    import numpy as np

    def preprocess_image(image_path):
        img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg16.preprocess_input(img)
        return img


    preprocessed_image = preprocess_image(filename)

    predictions = model.predict(preprocessed_image)

    from tensorflow.keras.applications.vgg16 import decode_predictions

    decoded_predictions = decode_predictions(predictions, top=1)
    predicted_label = decoded_predictions[0][0][1]
    return f"File {filename[8:]} processed successfully\nPredicted label: {predicted_label}"
    

if __name__ == '__main__':
    app.run()
