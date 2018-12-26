from keras.applications import MobileNetV2, InceptionV3
import pickle
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input, Dropout, MaxPooling2D
from keras.models import Model, load_model
from keras import callbacks, optimizers
import glob
import random


image_size = 224

path = r'C:\Users\trist\Documents\whale/'

img_aug_prob = .5


def random_crop(img):
    width = random.randint(int(img.shape[1]/2), img.shape[1])
    height = random.randint(int(img.shape[0]/2), img.shape[0])

    x = random.randint(0, img.shape[1]-width)
    y = random.randint(0,  img.shape[0]-height)
    img = img[y:y+height, x:x+width]
    return img


def random_img_modification(np_img):
    # if random.random() < img_aug_prob:
    #     np_img = random_crop(np_img)

    if random.random() < img_aug_prob:
        np_img = np.flip(np_img, 0)
    if random.random() < img_aug_prob:
        np_img = np.flip(np_img, 1)
    return np_img



def get_net2():
    base_model = MobileNetV2(include_top=False, input_shape=(image_size, image_size, 3))
    x = Flatten()(base_model.output)
    model = Model(input = base_model.input, output = x)
    adam = optimizers.Adam(lr=.0001, decay=1e-5)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    return model


def process_image(image_location):
    # start_image = np.array(Image.open(image_location).convert('LA'))[:,:,0]
    start_image = Image.open(image_location)
    start_image = start_image.resize((image_size, image_size))

    n_dims = len(np.array(start_image).shape)
    if n_dims == 3:
        start_image = np.array(start_image)[:, :, 0:3]
    else:
        new_img = []
        new_img.append(np.array(start_image))
        new_img.append(np.array(start_image))
        new_img.append(np.array(start_image))
        start_image = np.stack(new_img, axis=2)
    start_image = start_image.astype(np.float64)
    start_image /= 255.0
    start_image = random_img_modification(start_image)
    return start_image


def predict():
    test_files = glob.glob(path + 'test/*')[:10]
    file_names = []
    x_pred = []
    for i in test_files:
        file_names.append(i.split('/')[-1])
        x_pred.append(process_image(i))

    x_pred = np.array(x_pred)
    # x_pred = np.expand_dims(x_pred, 3)

    model = get_net2()
    preds = model.predict(x_pred)

    with open(path + 'le.pkl', 'rb') as f:
        le = pickle.load(f)

    for i, j in zip(file_names, preds):
        j[0] = 0
        pred_index = np.argmax(j)
        print(pred_index, le.inverse_transform([pred_index]))




if __name__ == '__main__':
    # train()
    predict()