from tensorflow.keras import Sequential
from tensorflow.keras.layers import *


############################################
kernel_size = (3, 3)
pool_size = (2, 2)
strides = (1, 1)
input_shape = (28, 28, 1)
############################################

def digit_model():
    model = Sequential([

        # First Layer of Convolution
        Conv2D(32, kernel_size=kernel_size, padding="valid", strides=strides, activation="relu",
               input_shape=input_shape),
        BatchNormalization(),
        MaxPool2D(pool_size=pool_size),

        # Second Layer of Convolution
        Conv2D(64, kernel_size=kernel_size, padding="valid", strides=strides, activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=pool_size),

        # Flattening the Layer
        Flatten(),

        # Adding the Dropout Layer
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(10, activation="softmax")
    ])

    # Printing the summary of the model
    print(model.summary())

    # Compiling the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model