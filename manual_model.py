import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import matplotlib.pyplot as plt
import PIL
import numpy as np
import os
import sys
from keras.utils import image_dataset_from_directory
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, array_to_img
from skimage.transform import resize  # Assuming you have scikit-image installed


class Model:
    def __init__(self, filename: str = None, lr = 0.0001):
        if filename is not None:
            self.model = keras.models.load_model(filename, lr)
        else:
            self.model = self.make_model()

    def make_model(self, file:str = None, lr  = 0.0001) -> keras.Sequential:
        if file is not None:
            self.model = keras.models.load_model(file)
            return self.model

        model = keras.Sequential()

        # INPUT:
        model.add(keras.layers.Normalization(input_shape=(28, 28, 1)))
        # CONV 1:
        model.add(keras.layers.Conv2D(32, (3, 3), padding="same", activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.2))
        # CONV 2:
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
        # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.2))
        # CONV 3:
        model.add(keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.2))
        # DENSE:
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        # OUTPUT:
        model.add(keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        return model
    
    def augment(self, ds):
        # # Augment the data:
        # horiz = ds.map(self.flip_hori)
        # vert = ds.map(self.flip_vert)
        rot = ds.map(self.rotate)
        # # Concatenate:
        # ds = ds.concatenate(horiz)
        # ds = ds.concatenate(vert)
        ds = ds.concatenate(rot)
        return ds

    # flip both image and mask identically
    def flip_hori(self, img, mask):
        img = tf.image.flip_left_right(img)
        return img, mask

    # flip both image and mask identically
    def flip_vert(self, img, mask):
        img = tf.image.flip_up_down(img)
        return img, mask

    # rotate both image and mask identically
    def rotate(self, img, mask):
        img = tf.image.rot90(img)
        return img, mask
    
    def load_data(self, directory: str):
        # Define the image size and batch size
        img_size = (28, 28)

        dataset = image_dataset_from_directory(directory, image_size=img_size, color_mode="grayscale")
        # Split them using sklearn
        print("Splitting data...")

        # Convert the tf.data.Dataset to NumPy arrays
        images = []
        labels = []

        for image_batch, label_batch in dataset:
            # Iterate through the batch and resize each image
            resized_images = [resize(img_to_array(img), img_size, mode='reflect') for img in image_batch.numpy()]
            
            images.extend(resized_images)
            labels.extend(label_batch.numpy())

        images = np.array(images)
        labels = np.array(labels)
        self.images = images
        self.labels = labels

        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def knn(self, input_image):
        # KNN:
        # Find the distance between the input image and all other images in the dataset
        distances = np.sqrt(np.sum(np.square(self.images - input_image), axis=(1, 2, 3)))
        print(distances)
        # Find the image with the minimum distance
        min_index = np.argmin(distances)
        print(self.labels[min_index])
        # Return the label of the nearest image
        return self.labels[min_index]


    

if __name__ == "__main__":
    print("Making model...")
    # m = Model(filename = "manual_model.keras")
    m = Model(lr = 0.0001)

    print("Loading data...")
    # Load data as a label from each of the folders 0-9 (4.1 INCLUDED):
    data_dir = './custom_dataset/'
    X_train, X_test, y_train, y_test = Model.load_data(data_dir)
    # Data augmnentation:
    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    ds_train = m.augment(ds_train)
    ds_test = m.augment(ds_test)
    # Pre-training:
    batch_size = 32
    steps_per_epoch = 100
    epochs = 100
    repeats = steps_per_epoch * epochs

    ds_train = ds_train.cache().shuffle(1024).batch(batch_size).repeat()
    plt.figure(figsize=(10, 10))
    for images, labels in ds_train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()

    ds_test = ds_test.cache().shuffle(1024).batch(batch_size).repeat()

    print("Training model...")
    m.model.fit(
        ds_train, 
        validation_data=(X_test, y_test),
        epochs=100, 
        steps_per_epoch=100)
    
    # Save Model:
    m.model.save("manual_model.keras")

    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]





