from model import digit_model
import tensorflow as tf
import matplotlib.pyplot as plt
####################################################
epochs = 10
####################################################

def learning_curve(train_val, test_val, title, xlabel, ylabel):
    plt.figure()
    plt.plot(train_val)
    plt.plot(test_val)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Loading the Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Reshaping the Model
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Encoding the Labels to categorical
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Normalizing the Data
X_train = X_train/255.0
X_test = X_test/255.0

# Loading the Compiled Model
model = digit_model()

# Training the Model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=100, validation_data=(X_test, y_test), verbose=1)

# Saving the Model
model.save('models/final_model.h5')

# plotting the Learning curve
learning_curve(history.history["accuracy"], history.history["val_accuracy"], "Accuracy curve", "Epochs", "Accuracy")
learning_curve(history.history["loss"], history.history["val_loss"], "Loss curve", "Epochs", "Loss")
