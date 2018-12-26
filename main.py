

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


def random_rotate(img):
    rn = random.randint(1,4)
    img = img.rotate(rn*90)
    return img


def transpose(img):
    img = img.transpose(method =Image.TRANSPOSE)
    return img


def random_img_modification(img):
    if random.random() < img_aug_prob:
        img = transpose(img)
    if random.random() < img_aug_prob:
        img = random_rotate(img)
    return img


def process_image(image_location, training_img = False):
    # start_image = np.array(Image.open(image_location).convert('LA'))[:,:,0]
    start_image = Image.open(image_location)
    if training_img:
        start_image = random_img_modification(start_image)
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
    # start_image = random_img_modification(start_image)
    return start_image


def gen_balanced_data(df, max_val, img_base_path, batch_size=32, training_set = True, balance_classes = False):
    x, y = [], []

    while True:
        if training_set and balance_classes:
            targets = list(df['Id'])
            next_target = random.choice(targets)
            sample_df = df[df['Id'] == next_target]
            image = random.choice(list(sample_df['Image']))
            next_x = process_image(img_base_path + image, training_img=True)
            new_y = [0 for _ in range(max_val + 1)]
            ids_df = sample_df[sample_df['Image'] == image]
            ids = ids_df['Id'].tolist()
            new_y[ids[0]] = 1
        else:
            images = list(df['Image'])
            next_image = random.choice(images)
            sample_df = df[df['Image'] == next_image]

            next_x = process_image(img_base_path + next_image, training_img=True)
            new_y = [0 for _ in range(max_val + 1)]

            ids = sample_df['Id'].tolist()
            new_y[ids[0]] = 1

        x.append(next_x)
        y.append(new_y)

        if len(x) == batch_size:
            x = np.array(x)
            y = np.array(y)
            yield x, y
            x, y = [], []


def train():
    df = pd.read_csv(path + 'train.csv')
    df['Id'] = df['Id'].apply(lambda x: '1' if 'new_' in x else x)
    df = df[df['Id'] != '1']

    # df2 = df['Id'].value_counts()
    le = LabelEncoder()
    df['Id'] = le.fit_transform(df['Id'])

    with open(path + 'le.pkl', 'wb') as f:
        pickle.dump(le, f)

    max_val = max(df['Id'])
    train_data, val_data = train_test_split(df, test_size=.05)

    def to_train_encoded_x_y(df, max_val):
        x, y = [], []
        for i, j in df.iterrows():
            print(i)
            new_y = [0 for _ in range(max_val + 1)]
            new_y[j['Id']] = 1
            image_location = path  + 'train/'+ j['Image']
            new_x = process_image(image_location)
            x.append(new_x)
            y.append(new_y)
        return np.array(x), np.array(y)



    def get_net2(y_size):
        print(y_size)
        model = MobileNetV2( classes=y_size + 1, weights=None)
        adam = optimizers.Adam(lr=.0001, decay=1e-5)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        return model


    # train_x, train_y = to_train_encoded_x_y(train_data, max_val)
    # val_x, val_y = to_train_encoded_x_y(val_data, max_val)
    # train_x = np.expand_dims(train_x, 3)
    # val_x = np.expand_dims(val_x, 3)

    cb = callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=10,
                                      verbose=0, mode='auto')
    # reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=0, verbose=1, min_lr=.000001)
    mcp_save = callbacks.ModelCheckpoint(path + 'model_no_new.h5', save_best_only=True, monitor='val_loss', verbose=1)

    # net = get_net(max_val)
    net = get_net2(max_val)

    train_gen = gen_balanced_data(train_data, max_val, path + 'train/', batch_size=32, training_set=True)
    val_gen = gen_balanced_data(val_data, max_val, path + 'train/', batch_size=32, training_set=False)
    net.fit_generator(train_gen, validation_data=val_gen, epochs=100, callbacks=[cb, mcp_save], steps_per_epoch=100, validation_steps=10)


def predict():
    test_files = glob.glob(path + 'test/*')[:100]
    file_names = []
    x_pred = []
    for i in test_files:
        file_names.append(i.split('/')[-1])
        x_pred.append(process_image(i))

    x_pred = np.array(x_pred)
    # x_pred = np.expand_dims(x_pred, 3)

    model = load_model(path + 'model.h5')
    preds = model.predict(x_pred)

    with open(path + 'le.pkl', 'rb') as f:
        le = pickle.load(f)

    output_dicts = []
    for i, j in zip(file_names, preds):
        j[0] = 0
        pred_index = np.argmax(j)
        print(pred_index, le.inverse_transform([pred_index]))



if __name__ == '__main__':
    train()
    # predict()





