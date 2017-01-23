
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Convolution2D
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


# command line flags
data_path = './training/'
val_path = './val/'
epochs = 30
batch_size = 1024

training_size = 30000
validation_size = 10000

dropout = .30

def load_image(filename, log_path):
    #print(log_path+filename)

    return cv2.cvtColor(cv2.imread(log_path+str.strip(filename)),cv2.COLOR_BGR2RGB)

# crop camera image to fit nvidia model input shape
def crop_camera(img, crop_height=66, crop_width=200):
    height = img.shape[0]
    width = img.shape[1]

    # y_start = 60+random.randint(-10, 10)
    # x_start = int(width/2)-int(crop_width/2)+random.randint(-40, 40)
    y_start = 60
    x_start = int(width/2)-int(crop_width/2)

    return img[y_start:y_start+crop_height, x_start:x_start+crop_width]

# if driving in a straight line remove extra rows
def filter_driving_straight(data_df, hist_items=5):
    print('filtering straight line driving with %d frames consective' %
          hist_items)
    steering_history = deque([])
    drop_rows = []

    for idx, row in data_df.iterrows():
        # controls = [getattr(row, control) for control in vehicle_controls]
        #print(row[3])
        #steering = getattr(row, 'steering')
        steering = row[3]

        # record the recent steering history
        steering_history.append(steering)
        if len(steering_history) > hist_items:
            steering_history.popleft()

        # if just driving in a straight
        if steering_history.count(0.0) == hist_items:
            drop_rows.append(idx)

    # return the dataframe minus straight lines that met criteria
    return data_df.drop(data_df.index[drop_rows])


def data_generator(path=data_path, batch_size=128):
    filter_straights =True
    log = pd.read_csv(path+'driving_log.csv')
    if filter_straights:
        log = filter_driving_straight(log)

    #X = log.ix[:,0]
    X=pd.concat([log.ix[:,0], log.ix[:,1], log.ix[:,2]])
    y= pd.concat([log.ix[:,3], log.ix[:,3]+0.25, log.ix[:,3]-0.25])
    #X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.33, random_state = 42)
    data_count = len(X)

    print("Loading training data with %d rows." % (data_count))

    while True:  # need to keep generating data

        # initialise data extract
        features = []
        labels = []

        # create a random batch to return
        while len(features) < batch_size:
            row = np.random.randint(data_count - 1)

            image = crop_camera(load_image(X.iloc[row], path))
            steering = y.iloc[row]

            features.append(image)
            labels.append(steering)

        # yield the batch
        yield (np.array(features), np.array(labels))

def val_generator(path=val_path, batch_size=128):

    log = pd.read_csv(path+'/driving_log.csv')
    #X = log.ix[:,0]
    X=pd.concat([log.ix[:,0], log.ix[:,1], log.ix[:,2]])
    y= pd.concat([log.ix[:,3], log.ix[:,3]+0.25, log.ix[:,3]-0.25])
    #X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.33, random_state = 42)
    data_count = len(X)

    print("Loading Validation data with %d rows." % (data_count))

    while True:  # need to keep generating data

        # initialise data extract
        features = []
        labels = []

        # create a random batch to return
        while len(features) < batch_size:
            row = np.random.randint(data_count - 1)

            image = crop_camera(load_image(X.iloc[row], path))
            steering = y.iloc[row]

            features.append(image)
            labels.append(steering)

        # yield the batch
        yield (np.array(features), np.array(labels))


def build_nvidia_model(img_height=66, img_width=200, img_channels=3,
                       dropout=.4):

    # build sequential model
    model = Sequential()

    # normalisation layer
    img_shape = (img_height, img_width, img_channels)
    model.add(Lambda(lambda x: x * 1./127.5 - 1,
                     input_shape=(img_shape),
                     output_shape=(img_shape), name='Normalization'))

    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

    for l in range(len(nb_filters)):
        model.add(Convolution2D(nb_filters[l],
                                kernel_size[l][0], kernel_size[l][1],
                                border_mode=padding[l],
                                subsample=strides[l],
                                activation='elu'))
        model.add(Dropout(dropout))

    # flatten layer
    model.add(Flatten())

    # fully connected layers with dropout
    neurons = [100, 50, 10]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l], activation='elu'))
        model.add(Dropout(dropout))

    # logit output - steering angle
    model.add(Dense(1, activation='elu', name='Out'))

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='mse')
    return model

def get_callbacks():
     checkpoint = ModelCheckpoint(
         "checkpoints/model-{val_loss:.4f}.h5",
         monitor='val_loss', verbose=1, save_weights_only=True,
         save_best_only=True)

     tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                               write_graph=True, write_images=False)

     return [checkpoint, tensorboard]

    #earlystopping = EarlyStopping(monitor='val_loss', min_delta=0,
    #                              patience=1, verbose=1, mode='auto')
    # return [earlystopping, checkpoint]
    #return [earlystopping]



if __name__ == '__main__':

    # build model and display layers
    model = build_nvidia_model(dropout=dropout)
    print(model.summary())

    plot(model, to_file='model.png', show_shapes=True)

    model.fit_generator(data_generator(data_path, batch_size),
        samples_per_epoch=training_size,
        nb_epoch=epochs,
        callbacks=get_callbacks(),
        validation_data=val_generator(val_path, batch_size),
        nb_val_samples=validation_size)

    # save weights and model
    model.save_weights('model.h5')
    with open('model.json', 'w') as modelfile:
        json.dump(model.to_json(), modelfile)