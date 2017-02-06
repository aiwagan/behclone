import json
import random
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import PReLU
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Program Parameters
data_path = './data/'
val_path = './data/'
epochs = 20
batch_size = 256
dropout = .20

# Driving log file is loaded here.
log = pd.read_csv(data_path + 'driving_log.csv')

# Angle Filtering algorithm. We break the log into separate lists for each set of angles from -1 to 1 with a step size of 0.1.
kl = []
for i in np.arange(-1.0, 1.0, 0.1):
    kl.append(log[(log['steering'] > i) & (log['steering'] < (i + 0.1))])
df = kl[0]

# Here we count the angles which are comparatively larger in number (2000) and filter these to arounf 700 values
for i in range(1, len(kl)):
    if (kl[i].size < 2000):
        df = pd.concat([df, kl[i]])
    else:
        df = pd.concat([df, kl[i].iloc[0:700, :]])

log = df

# Driving log data contained three center, left and right images. Here it is separated and concatenated to add edge cases. 
# 0.3 is added to the steering angle for left side images and 0.3 is subtracted from the right side images.
X = pd.concat([log.ix[:, 0], log.ix[:, 1], log.ix[:, 2]])
y = pd.concat([log.ix[:, 3], log.ix[:, 3] + 0.3, log.ix[:, 3] - 0.3])

# Break the driving log into Training(80%) and Testing(20%) set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

num_train_images = len(X_train)
num_test_images = len(X_test)




def load_image(filename, log_path):
    """
        :param filename: Name of the file to load
        :param log_path: Name of the directory
        :return: return an image in RGB color Space
    """
    return cv2.cvtColor(cv2.imread(log_path + str.strip(filename)), cv2.COLOR_BGR2RGB)


def random_shadow(img):
    """
        :param img: An image 
        :return: return an image
        Algorirthm randomly generates a rectangle which starts from the top of the image to the bottom of the image
        The rectangle positions are merged to form a mask image which is subtracted from the input image to generate an image containing shadows. 
    """
    random.seed()

    rect = [[random.randint(0, 100), 0], [random.randint(100, 200), 0], [random.randint(100, 200), 66],
            [random.randint(0, 100), 66]]
    poly = np.array([rect], dtype=np.int32)
    imgroi = np.zeros((66, 200), np.uint8)
    cv2.fillPoly(imgroi, poly, np.random.randint(50, 100))
    img3 = cv2.merge([imgroi, imgroi, imgroi])

    dst = cv2.subtract(img, img3)
    return dst


# brightness - referenced Vivek Yadav post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0

def randomise_image_brightness(image):
    """
        :param image: Input image
        :return: return an image in RGB Color space with randimly modified image brightness.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bv = .4 + np.random.uniform()
    hsv[::2] = hsv[::2] * bv

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# referenced Vivek Yadav post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
def jitter_image_rotation(image, steering):
    """
    :param image: input image 
    :param steering: steering angle
    :return: return an image which is randomlyand an steering angle randomly rotated the angle is modfied according to the rotation.
    """

    rows, cols, _ = image.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange / 2
    steering = steering + transX / transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels / 2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))
    return image, steering



# crop camera image to fit nvidia model input shape
def crop_camera(img, crop_height=66, crop_width=200):
    """
    :param img: Input image data
    :param crop_height: cropped height required
    :param crop_width: returned width required
    :return: return an image with the desired height and width
    """
    height = img.shape[0]
    width = img.shape[1]
    y_start = 60
    x_start = int(width / 2) - int(crop_width / 2)
    return img[y_start:y_start + crop_height, x_start:x_start + crop_width]


def data_generator(path=data_path, batch_sz=128):
    """
    :param path: Path to the data only pass the parent folder
    :param batch_sz: Batch size required to be returned by this generator
    :return: return a list of images and steering angles
    """
    train_batch_pointer = 0
    while True:
        features = []
        labels = []
        for i in range(0, batch_sz):
            row = (train_batch_pointer + i) % num_train_images
            img = crop_camera(load_image(X_train.iloc[row], path))
            steering = y_train.iloc[row]
            # Image augmentation is performed here. I choose to use random_shadow, rotation and flip as part of my augmentations.
            image = random_shadow(img)
            image, steering = jitter_image_rotation(image, steering)
            # flip 50% randomily that are not driving straight
            if random.random() >= .5:  # and abs(steering) > 0.1:
                image = cv2.flip(image, 1)
                steering = -steering

            features.append(image)
            labels.append(steering)

        train_batch_pointer += batch_sz

        yield (np.array(features), np.array(labels))


def val_generator(path=val_path, batch_sz=128):
    """
    :param path: Path to the data only pass the parent folder
    :param batch_sz: Batch size required to be returned by this generator
    :return: return a list of image and steering angles
    """
    val_batch_pointer = 0
    while True:
        features = []
        labels = []

        for i in range(0, batch_sz):
            # print('test='+val_batch_pointer)
            row = (val_batch_pointer + i) % num_test_images
            features.append(crop_camera(load_image(X_test.iloc[row], path)))
            labels.append(y_test.iloc[row])

        val_batch_pointer += batch_sz
        yield (np.array(features), np.array(labels))


def build_nvidia_model(img_height=66, img_width=200, img_channels=3, dropout=.4):
    # build sequential model
    model = Sequential()

    # normalisation layer
    img_shape = (img_height, img_width, img_channels)
    model.add(
        Lambda(lambda x: (x - 127.5) / 255, input_shape=(img_shape), output_shape=(img_shape), name='Normalization'))

    model.add(Convolution2D(32, 5, 5, border_mode='valid', subsample=(2, 2)))  # , activation='elu'))
    model.add(PReLU())
    model.add(Convolution2D(64, 5, 5, border_mode='valid', subsample=(2, 2)))  # , activation='prelu'))
    model.add(PReLU())
    model.add(Convolution2D(128, 5, 5, border_mode='valid', subsample=(2, 2)))  # , activation='prelu'))
    model.add(PReLU())
    # model.add(Dropout(dropout))

    model.add(Convolution2D(128, 3, 3, border_mode='valid', subsample=(1, 1)))  # , activation='prelu'))
    model.add(PReLU())
    model.add(Convolution2D(128, 3, 3, border_mode='valid', subsample=(1, 1)))  # , activation='prelu'))
    model.add(PReLU())
    model.add(Dropout(dropout))

    # flatten layer
    model.add(Flatten())

    # fully connected layers with dropout

    model.add(Dense(256))  # ,activation='prelu'))

    model.add(PReLU())
    model.add(Dropout(dropout))

    model.add(Dense(128))  # ,activation='prelu'))
    model.add(PReLU())
    model.add(Dropout(dropout))

    model.add(Dense(64))  # ,activation='prelu'))
    model.add(PReLU())
    model.add(Dropout(dropout))

    model.add(Dense(1, name='OutputAngle'))  # ,  activation='relu', name='Out'\
    model.compile(optimizer='nadam', loss='mse')
    return model




# Start of the main function
if __name__ == '__main__':
    # build model and display layers
    model = build_nvidia_model(dropout=dropout)
    print(model.summary())

    plot(model, to_file='model.png', show_shapes=True)

    checkpoint = ModelCheckpoint("checkpoints/model-{val_loss:.4f}.h5",
                                 monitor='val_loss', verbose=1,
                                 save_weights_only=True, save_best_only=True)

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    model.fit_generator(data_generator(data_path, batch_size),
                        samples_per_epoch=batch_size * int(num_train_images / batch_size),
                        nb_epoch=epochs, callbacks=[earlystopping, checkpoint],
                        validation_data=val_generator(val_path, batch_size),
                        nb_val_samples=num_test_images)

    # save weights and model
    model.save_weights('model.h5')
    with open('model.json', 'w') as modelfile:
        json.dump(model.to_json(), modelfile)
