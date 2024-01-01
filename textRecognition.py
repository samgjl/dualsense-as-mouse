import numpy as np
# import pandas as pd
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import matplotlib.pyplot as plt

ON = 255
OFF = 0

class TextRecognition:
    def __init__(self, width = 1920, height = 1080):
        self.width = width
        self.height = height
        # Arrays:
        self.positions = [] # remains 1D array
        self.matrix = [] # becomes 2D array
        # Tracking:
        self.isActive = False
        self.tracking = False
        # Classifiers:
        self.digits = Model("digits")
        # self.digits.load("manual_model.keras")
        self.digits.load("digits.keras")

        self.letters = Model("letters")
        self.letters.load("letters.keras")

        self.mode = "digits"
    
    # Add a point to the array of positions
    # @PARAMS: point - tuple of x and y coordinates
    def addPoint(self, point: tuple) -> None:
        # inter-frame interpolation:
        if len(self.positions) > 0:
            previousPoint = self.positions[-1]
            self.positions.append(((point[0]+previousPoint[0])/2, (point[1]+previousPoint[1])/2))
        # add point:
        self.positions.append(point)

    # Create a numpy matrix from an array of positions
    # @PARAMS: array - 1D array of tuples
    def matrixFromPositions(self):        
        # TODO: size = 28 # We're using the MNIST dataset, which is 28x28
        size = 28
        self.matrix = np.zeros((size, size), dtype=np.uint8)
        # Update positions:
        for index in self.positions:
            x = int(index[0]*(size/self.width))
            y = int(index[1]*(size/self.height))
            self.matrix[y, x] = ON # Form y-coordinate, x-coordinate (row by column)
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    if (x+i >= 0 and x+i < size) and (y+j >= 0 and y+j < size):
                        self.matrix[y+j, x+i] = min(max(self.matrix[y+j, x+i], self.matrix[y+j, x+i] + (ON/27)), ON)
                        pass

        self.positions = [] # Reset positions
        return self.matrix

    def predict(self, array2D: np.ndarray = None) -> str:
        if array2D is None:
            array2D = self.matrix

        if self.mode == "digits":
            return self.digits.predict(array2D)
        elif self.mode == "letters":
            return self.letters.predict(array2D)

    def savePNG(self, array2D: np.array = None, filename: str = "output.png") -> None:
        if array2D is None:
            array2D = self.matrix
        if filename[-4:] != ".png":
            filename += ".png"
        mpimg.imsave(filename, array2D, cmap='gray')

def preprocess_image(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, (28, 28))
            return image, label

class Model():
    def __init__(self, classifier_type: str = "digits"):
        self.model = None # TODO: Load model from file
        self.type = classifier_type
        if self.type == "digits":
            self.labels = [str(i) for i in range(10)]
        elif self.type == "letters":
            self.labels = [chr(i) for i in range(65, 91)]
        
    def load(self, filename: str = None) -> None:
        if filename is None:
            if self.type == "digits":
                filename = "digits.keras"
            elif self.type == "letters":
                filename = "letters.keras"

        self.model = keras.models.load_model(filename)


    def build(self):
        # Define the model
        addon = 1 if self.type == "letters" else 0
        self.model = keras.Sequential([
            # CONV 1:
            keras.layers.Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, kernel_size = 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            # CONV 2:
            keras.layers.Conv2D(64, kernel_size = 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, kernel_size = 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            # FINAL CONV:
            keras.layers.Conv2D(128, kernel_size = 4, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.4),
            # DENSE:
            keras.layers.Dense(len(self.labels) + addon, activation='softmax')
        ])
        #* OLD: Too shallow?
        # self.model = keras.Sequential([
        #     keras.layers.Normalization(input_shape=(28, 28, 1)),
        #     keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(28, 28, 1)),
        #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #     keras.layers.Dropout(0.2),
        #     keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu'),
        #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #     keras.layers.Dropout(0.2),
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(128, activation='relu'),
        #     keras.layers.Dropout(0.1),
        #     keras.layers.Dense(len(self.labels) + addon, activation='softmax') # TODO: +1? Really?
        # ])

        # Compile the model
        self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    def train(self, epochs = 5, save: bool = False) -> None:
        # Load EMNIST dataset:
        if self.type == "digits":
            ds_train, ds_info = tfds.load('emnist/digits', split='train', shuffle_files=True, as_supervised=True, with_info=True)
            ds_test = tfds.load('emnist/digits', split='test', shuffle_files=True, as_supervised=True)
        elif self.type == "letters":
            ds_train, ds_info = tfds.load('emnist/letters', split='train', shuffle_files=True, as_supervised=True, with_info=True)
            ds_test = tfds.load('emnist/letters', split='test', shuffle_files=True, as_supervised=True) 
        
        # Randomize and preprocess the datasets:
        ds_train = ds_train.map(preprocess_image)
        ds_train = ds_train.shuffle(1024)
        # augment the dataset:
        # ds_train = ds_train.map(lambda image, label: (tf.image.random_flip_left_right(image), label))
        # ds_train = ds_train.map(lambda image, label: (tf.image.random_flip_up_down(image), label))
        # ds_train = ds_train.map(lambda image, label: (tf.image.random_crop(image, [28, 28, 1]), label))


        ds_test = ds_test.map(preprocess_image)
        ds_test = ds_test.shuffle(1024)
        # ds_test = ds_test.map(lambda image, label: (tf.image.random_flip_left_right(image), label))
        # ds_test = ds_test.map(lambda image, label: (tf.image.random_flip_up_down(image), label))
        # ds_test = ds_test.map(lambda image, label: (tf.image.random_crop(image, [28, 28, 1]), label))

        # Define the number of steps per epoch
        batch_size = 32
        steps_per_epoch = ds_info.splits['train'].num_examples // batch_size


        # Train the model using ds_train
        self.model.fit(
                       ds_train.batch(batch_size),
                       validation_data = ds_test.batch(batch_size), 
                       epochs=epochs, 
                       steps_per_epoch=steps_per_epoch,
                       )

        # Save the model:
        if save == True:
            if self.type == "digits":
                self.model.save("digits.keras")
            elif self.type == "letters":
                self.model.save("letters.keras")
        
    def predict(self, array2D: np.ndarray) -> str:
        # # Preprocess the image #* NEED????????????????????????????????????????
        # array2D = np.expand_dims(array2D, axis=0)
        # array2D = np.expand_dims(array2D, axis=3)
        array2D = array2D.astype('float32')
        array2D = array2D / 255.0
        array2D = array2D.reshape(1, 28, 28, 1)

        plt.imshow(array2D[0], cmap='gray')
        plt.show()

        # Predict:
        predictions = self.model.predict(array2D)
        print("Prediction: " + str((predictions)))


        remove = 1 if self.type == "letters" else 0
        output = self.labels[np.argmax(predictions[0]) - remove]
        print(output)
        return output # turn the probabilities into indices of our labels list.


if __name__ == "__main__":
    letters = Model(classifier_type="letters")
    # letters.load("letters.keras")
    letters.build()
    letters.train(epochs = 10, save = True)

    digits = Model(classifier_type="digits")
    # digits.load("digits.keras")
    digits.build()
    digits.train(epochs = 10, save = True)


    from PIL import Image
    image = Image.open("output.png")
    image = image.convert('L')
    image = np.array(image)

    print("\nLETTERS: \n")
    # letters.predict(image)
    print("\nDIGITS: \n")
    digits.predict(image)