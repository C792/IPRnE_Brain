from flask import Flask, request, render_template, send_from_directory, url_for
import os

app = Flask(__name__)

fname = ''
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global fname
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        fname = filename
        return vgg16process(filename)
    
@app.route('/uploaded/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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

    ans = model.predict(preprocessed_image)

    d_name = [
        'pituitary',
        'notumor',
        'meningioma',
        'glioma',
    ]
    for i in range(4):
        if ans[0][i]:
            return f"""File {filename[8:]} processed successfully
            <br>
            View your image <a href="{url_for("uploaded_file", filename=filename[8:])}">here</a>
            <br>
            Predicted label: { d_name[i] }"""

if __name__ == '__main__':
    app.run()
