import numpy as np
# import pandas as pd
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

ON = 1
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
        self.classifier = Model()
        self.classifier.load("model.keras")
    
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
        # Create a matrix of zeros (rows-by-columns):
        size = 28 # We're using the MNIST dataset, which is 28x28
        self.matrix = np.zeros((size, size), dtype=np.uint8)
        # Update positions:
        for index in self.positions:
            x = int(index[0]*(size/self.width))
            y = int(index[1]*(size/self.height))
            # TODO: Add delocalization or remove this. Put up or shut up.
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    if (x+i >= 0 and x+i < size) and (y+j >= 0 and y+j < size):
                        self.matrix[y+j, x+i] = min(self.matrix[y+j, x+i] + (ON/10), ON)

            self.matrix[y, x] = 1 # Form y-coordinate, x-coordinate (row by column)
        
        self.positions = [] # Reset positions
        return self.matrix

    def savePNG(self, array2D: np.ndarray, filename: str = "output.png") -> None:
        if filename[-4:] != ".png":
            filename += ".png"
        mpimg.imsave(filename, array2D)

def preprocess_image(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, (28, 28))
            return image, label

class Model():
    def __init__(self):
        self.model = None # TODO: Load model from file
        self.labels = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)] + list('abdefghnqrt') # either 0-9 or A-Z

    def load(self, filename: str = "model.keras") -> None:
        self.model = keras.models.load_model(filename)


    def build(self):
        # Define the model
        self.model = keras.Sequential([
            keras.layers.Normalization(),
            keras.layers.Flatten(input_shape=(28, 28, 1)),  # Flatten the 28x28x1 images to a 784-element vector
            keras.layers.Dense(128, activation='relu'),
            # keras.layers.Dropout(0.2),
            keras.layers.Dense(len(self.labels), activation='softmax')
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    def train(self, epochs = 5) -> None:
        # Load EMNIST dataset with the 'balanced' split
        ds_train, ds_info = tfds.load('emnist/balanced', split='train', shuffle_files=True, as_supervised=True, with_info=True)
        ds_test = tfds.load('emnist/balanced', split='test', shuffle_files=True, as_supervised=True)
        # Preprocess the dataset

        # Randomize and preprocess the datasets:
        ds_train = ds_train.map(preprocess_image)
        ds_train = ds_train.shuffle(1024)
        ds_test = ds_test.map(preprocess_image)
        ds_test = ds_test.shuffle(1024)

        # Define the number of steps per epoch
        batch_size = 32
        steps_per_epoch = ds_info.splits['train'].num_examples // batch_size

        # Train the model using ds_train
        self.model.fit(ds_train.batch(batch_size), validation_data=ds_test.batch(batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch)

        # Save the model:
        self.model.save("model.keras")

        
    def predict(self, array2D: np.ndarray) -> str:
        predictions = self.model.predict(array2D)
        output = self.labels[np.argmax(predictions[0])]
        print(output)
        return output # turn the probabilities into indices of our labels list.


if __name__ == "__main__":
    model = Model()
    # model.build()
    model.load()
    # model.train()

    from PIL import Image
    image = Image.open("output.png")
    image = image.convert('L')
    image = np.array(image)
    model.predict(image)