import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt 
import numpy as np 
import json

model = load_model('C:/Users/sayea/Documents/express-project/mycontacts-backend/python/vgg16-transfer-learning-final.h5')

def crop_and_resize(image, target_w=224, target_h=224):
    '''this function crop & resize images to target size by keeping aspect ratio'''
    if image.ndim == 2:
        img_h, img_w = image.shape             # for Grayscale will be   img_h, img_w = img.shape
    elif image.ndim == 3:
        img_h, img_w, channels = image.shape   # for RGB will be   img_h, img_w, channels = img.shape
    target_aspect_ratio = target_w/target_h
    input_aspect_ratio = img_w/img_h

    if input_aspect_ratio > target_aspect_ratio:
        resize_w = int(input_aspect_ratio*target_h)
        resize_h = target_h
        img = cv2.resize(image, (resize_w , resize_h))
        crop_left = int((resize_w - target_w)/2)  ## crop left/right equally
        crop_right = crop_left + target_w
        new_img = img[:, crop_left:crop_right]
    if input_aspect_ratio < target_aspect_ratio:
        resize_w = target_w
        resize_h = int(target_w/input_aspect_ratio)
        img = cv2.resize(image, (resize_w , resize_h))
        crop_top = int((resize_h - target_h)/4)   ## crop the top by 1/4 and bottom by 3/4 -- can be changed
        crop_bottom = crop_top + target_h
        new_img = img[crop_top:crop_bottom, :]
    if input_aspect_ratio == target_aspect_ratio:
        new_img = cv2.resize(image, (target_w, target_h))

    return new_img


detector = MTCNN()  # creates detector  

def extract_face(img, target_size=(224,224)):
    '''this functions extract the face from different images by 
    1) finds the facial bounding box
    2) slightly expands top & bottom boundaries to include the whole face
    3) crop into a square shape
    4) resize to target image size for modelling
    5) if the facial bounding box in step 1 is not found, image will be cropped & resized to 224x224 square'''
           
    # 1. detect faces in an image
      
    results = detector.detect_faces(img)
    if results == []:    # if face is not detected, call function to crop & resize by keeping aspect ratio
        new_face = crop_and_resize(img, target_w=224, target_h=224)    
    else:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1+width, y1+height
        face = img[y1:y2, x1:x2]  # this is the face image from the bounding box before expanding bbox

        # 2. expand the top & bottom of bounding box by 10 pixels to ensure it captures the whole face
        adj_h = 30

        #assign value of new y1
        if y1-adj_h <10:
            new_y1=0
        else:
            new_y1 = y1-adj_h

        #assign value of new y2    
        if y1+height+adj_h < img.shape[0]:
            new_y2 = y1+height+adj_h
        else:
            new_y2 = img.shape[0]
        new_height = new_y2 - new_y1

        # 3. crop the image to a square image by setting the width = new_height and expand the box to new width
        adj_w = int((new_height-width)/2)    

        #assign value of new x1
        if x1-adj_w < 0:
            new_x1=0
        else:
            new_x1 = x1-adj_w

        #assign value of new x2
        if x2+adj_w > img.shape[1]:
            new_x2 = img.shape[1]
        else:
            new_x2 = x2+adj_w
        new_face = img[new_y1:new_y2, new_x1:new_x2]  # face-cropped square image based on original resolution

    # 4. resize image to the target pixel size
    sqr_img = cv2.resize(new_face, target_size)   
    return sqr_img


y_label_dict = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

def predict_face_shape(img_array):
    '''
    this function reads a single image in the form of an array, 
    and process the image then make predictions.
    '''
    try:
        # first extract the face using bounding box
        face_img = extract_face(img_array)  # call function to extract face with bounding box
        new_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) # convert to RGB -- use this for display          
        # convert the image for modelling
        test_img = np.array(new_img, dtype=float)
        test_img = test_img / 255
        test_img = np.array(test_img).reshape(1, 224, 224, 3)  
        # make predictions
        pred = model.predict(test_img)
        probabilities = np.round(pred.flatten() * 100, 2)
        prediction = {y_label_dict[i]: round(float(probabilities[i]), 2) for i in range(len(y_label_dict))}
        result = max(prediction, key=prediction.get)
        prediction['result'] = result
        print(prediction)
        return prediction

    except Exception as e:
        print(f'Oops! Something went wrong. Please try again.')
        return {'error': str(e)}



# Create the Flask app
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({
        "hello":"world"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        if image_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

            recommendation = {}

            face_shape = predict_face_shape(image)
            print(face_shape)

            if face_shape['result'] in ['Heart', 'Square', 'Oval', 'Oblong', 'Round']:
                file_name = face_shape['result']
                file_path = f'{file_name.lower()}.json'
                try:
                    with open(file_path, 'r') as file:
                        recommendation = json.load(file)
                except Exception as e:
                    print(f"Error opening file: {e}")
            else:
                recommendation = {'message': 'No recommendation found for this face shape'}



            response = {
                'result': face_shape,
                'recommendation': recommendation
            }

            return jsonify(response)

        else:
            return jsonify({'error': 'Invalid image format. Supported formats: PNG, JPG, JPEG'}), 400

    else:
        return jsonify({'error': 'Method not allowed. Use POST'}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)