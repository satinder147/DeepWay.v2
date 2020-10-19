import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from models.unet_keras import Models, load_data


def train():
    w = 256
    h = 256
    c = 3

    mod = Models(w, h, c)
    auto_encoder = mod.arch3()
    auto_encoder.summary()
    x_data, y_data = load_data()
    x_data = np.array(x_data, dtype='float')/255.0
    y_data = np.array(y_data, dtype='float')/255.0
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.1, random_state=30)
    auto_encoder.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    img_data_gen = ImageDataGenerator(**data_gen_args)
    img_data_gen.fit(train_x, augment=True, seed=7)
    img_data_gen.fit(train_y, augment=True, seed=7)
    image_generator=img_data_gen.flow(x=train_x, y=train_y, batch_size=32, shuffle=True)
    history = auto_encoder.fit_generator(image_generator, epochs=100,
                                         steps_per_epoch=3000, validation_data=(test_x, test_y), verbose=1)
    auto_encoder.save('road4.MODEL')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("res.png")


if __name__ == "__main__":
    train()


