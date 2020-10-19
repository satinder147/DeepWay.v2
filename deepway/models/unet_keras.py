import os
import cv2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D

orig = 'neworig'
seg = 'newseg'


def load_data():
    x_data = []
    y_data = []
    ori = os.listdir('/home/satinder/DeepWay.v2/dataSet/img')
    for i in ori:
        img = cv2.imread('/home/satinder/DeepWay.v2/dataSet/img'+'/'+i, 1)
        img = cv2.resize(img, (256, 256))
        x_data.append(img_to_array(img))
        img = cv2.imread('/home/satinder/DeepWay.v2/dataSet/mask'+'/'+i.split('.')[0]+".png", 1)
        img = cv2.Canny(img, 100, 150)
        img = cv2.dilate(img, None, iterations=5)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (256, 256))
        y_data.append(img_to_array(img))
    return x_data, y_data


class Models:
    def __init__(self, w, h, c):
        self.w = w
        self.h = h
        self.c = c

    def arch1(self):
        inp = Input(shape=(self.w, self.h, self.c))
        enc = Conv2D(64, (3, 3), padding='same')(inp)
        enc = BatchNormalization()(enc)
        enc = LeakyReLU(alpha=0.1)(enc)
        enc = MaxPooling2D(pool_size=(2, 2))(enc)
        enc = Conv2D(32, (3, 3), padding='same')(enc)
        enc = LeakyReLU(alpha=0.1)(enc)
        enc = BatchNormalization()(enc)
        enc = MaxPooling2D(pool_size=(2, 2))(enc)
        enc = Conv2D(16, (3, 3), padding='same')(enc)
        enc = LeakyReLU(alpha=0.1)(enc)
        enc = BatchNormalization()(enc)
        enc = MaxPooling2D(pool_size=(2, 2))(enc)
        enc = Conv2D(8, (3, 3), padding='same')(enc)
        enc = LeakyReLU(alpha=0.1)(enc)
        enc = MaxPooling2D(pool_size=(2, 2))(enc)

        dec = Conv2D(8, (3, 3), padding='same')(enc)
        dec = LeakyReLU(alpha=0.1)(dec)
        dec = UpSampling2D((2, 2))(dec)
        dec = Conv2D(16, (3, 3), padding='same')(dec)
        dec = LeakyReLU(alpha=0.1)(dec)
        dec = UpSampling2D((2, 2))(dec)
        dec = Conv2D(32, (3, 3), padding='same')(dec)
        dec = LeakyReLU(alpha=0.1)(dec)
        dec = UpSampling2D((2, 2))(dec)
        dec = Conv2D(64, (3, 3), padding='same')(dec)
        dec = LeakyReLU(alpha=0.1)(dec)
        dec = UpSampling2D((2, 2))(dec)
        final = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(dec)
        auto = Model(inp, final)
        return auto

    def arch2(self, input_size=(256, 256, 3)):
        inputs = Input(input_size)
        conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = Concatenate(axis=-1)([drop4, up6])
        conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = Concatenate(axis=-1)([up7, conv3])
        conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = Concatenate(axis=-1)([conv2,up8])
        conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = Concatenate(axis=-1)([conv1, up9])
        conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(3, 1, activation='sigmoid')(conv9)
        model = Model(input=inputs, output=conv10)
        model.summary()
        return model

    def arch3(self,input_size = (256,256,3)):
        inputs = Input(input_size)
        conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = Concatenate(axis=-1)([drop4,up6])
        conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = Concatenate(axis=-1)([up7,conv3])
        conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = Concatenate(axis=-1)([conv2,up8])
        conv8 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = Concatenate(axis=-1)([conv1, up9])
        conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(3, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)
        model.summary()
        return model
