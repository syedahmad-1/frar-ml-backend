import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import matplotlib.pyplot as plt

class FaceShapePredictor:
    def __init__(self, model_path, y_label_dict):
        self.model = load_model(model_path)
        self.y_label_dict = y_label_dict
        self.detector = MTCNN()

    def crop_and_resize(self, image, target_w=224, target_h=224):
        if image.ndim == 2:
            img_h, img_w = image.shape
        elif image.ndim == 3:
            img_h, img_w, channels = image.shape
        target_aspect_ratio = target_w / target_h
        input_aspect_ratio = img_w / img_h

        if input_aspect_ratio > target_aspect_ratio:
            resize_w = int(input_aspect_ratio * target_h)
            resize_h = target_h
            img = cv2.resize(image, (resize_w, resize_h))
            crop_left = int((resize_w - target_w) / 2)
            crop_right = crop_left + target_w
            new_img = img[:, crop_left:crop_right]
        elif input_aspect_ratio < target_aspect_ratio:
            resize_w = target_w
            resize_h = int(target_w / input_aspect_ratio)
            img = cv2.resize(image, (resize_w, resize_h))
            crop_top = int((resize_h - target_h) / 4)
            crop_bottom = crop_top + target_h
            new_img = img[crop_top:crop_bottom, :]
        else:
            new_img = cv2.resize(image, (target_w, target_h))

        return new_img

    def extract_face(self, img):
        results = self.detector.detect_faces(img)
        if results == []:
            new_face = self.crop_and_resize(img)
        else:
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height
            face = img[y1:y2, x1:x2]

            adj_h = 30
            if y1 - adj_h < 10:
                new_y1 = 0
            else:
                new_y1 = y1 - adj_h
            if y1 + height + adj_h < img.shape[0]:
                new_y2 = y1 + height + adj_h
            else:
                new_y2 = img.shape[0]
            new_height = new_y2 - new_y1

            adj_w = int((new_height - width) / 2)
            if x1 - adj_w < 0:
                new_x1 = 0
            else:
                new_x1 = x1 - adj_w
            if x2 + adj_w > img.shape[1]:
                new_x2 = img.shape[1]
            else:
                new_x2 = x2 + adj_w
            new_face = img[new_y1:new_y2, new_x1:new_x2]

        sqr_img = cv2.resize(new_face, (224, 224))
        return sqr_img

    def predict_face_shape(self, img_array):
        try:
            # face_img = self.extract_face(img_array)
            face_img = self.extract_face(img_array)
            new_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            test_img = np.array(new_img, dtype=float)
            test_img = test_img / 255
            test_img = np.array(test_img).reshape(1, 224, 224, 3)

            pred = self.model.predict(test_img)
            label = np.argmax(pred, axis=1)
            shape = self.y_label_dict[label[0]]
            print(f'Your face shape is {shape}')
            pred = np.max(pred)
            print(f'Probability {np.around(pred * 100, 2)}')
            plt.imshow(new_img)
            plt.show()
        except Exception as e:
            print(f'Oops!  Something went wrong.  Please try again.')


model_path = 'C:/Users/sayea/Documents/express-project/mycontacts-backend/python/vgg16-transfer-learning-final.h5'
y_label_dict = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}
face_shape_predictor = FaceShapePredictor(model_path, y_label_dict)

img_path = 'C:/Users/sayea/Documents/express-project/mycontacts-backend/uploads/1715065197830-image-c.jpg'
img = cv2.imread(img_path)

face_shape_predictor.predict_face_shape(img)