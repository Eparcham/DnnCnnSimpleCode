import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random
from tensorflow.keras.datasets import mnist
from tensorflow import keras

## mange gpu memory
if tf.__version__.startswith('2'):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

## load dataset and show ranom image
(X_train, y_train),(X_test, y_test) = mnist.load_data()
i = random.randint(1,60000) # select any random index from 1 to 60,000
plt.imshow( X_train[i] , cmap = 'gray') # reshape and plot the image
plt.show()

# Let's view more images in a grid format
W_grid = 15
L_grid = 15
fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))
axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array
n_training = len(X_train) # get the length of the training dataset

for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables
    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index
    axes[i].imshow( X_train[index] )
    axes[i].set_title(y_train[index], fontsize = 8)
    axes[i].axis('off')
plt.subplots_adjust(hspace=0.4)
plt.show()

## use shuffle for best performance in train
X_train, y_train = shuffle(X_train, y_train)

# Let's normalize the data
X_train = X_train / 255
X_test = X_test / 255

## sperat validation dataset from train data
ValidationPersent = 20
N_trainData = len(X_train)
N_validData = int((N_trainData*ValidationPersent)/100)
x_validation,y_validation = X_train[0:N_validData],y_train[0:N_validData]
## remove validation data from training data
X_train = X_train[N_validData+1:]
y_train = y_train[N_validData+1:]

## add 1 chanel in last in image
X_train = X_train[..., np.newaxis]
x_validation = x_validation[...,np.newaxis]
X_test = X_test[...,np.newaxis]

## create best optimizer
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

## Let's build the CNN
Cnn = tf.keras.models.Sequential()
Cnn.add(tf.keras.layers.Conv2D(16, (3,3), strides=1, padding="same", input_shape=(28, 28, 1)))
Cnn.add(tf.keras.layers.MaxPooling2D((2,2), padding="same"))
Cnn.add(tf.keras.layers.Conv2D(8, (3,3), strides=1, padding="same"))
Cnn.add(tf.keras.layers.MaxPooling2D((2,2), padding="same"))
Cnn.add(tf.keras.layers.Conv2D(8, (3,3), strides=1, padding="same"))
Cnn.add(tf.keras.layers.AveragePooling2D())
Cnn.add(tf.keras.layers.Flatten())
Cnn.add(tf.keras.layers.Dense(10,activation='softmax'))
Cnn.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
Cnn.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1,
    )
]

epoch_hist = Cnn.fit(X_train,y_train,
                     epochs=100,
                     batch_size = 128,
                     validation_data = (x_validation,y_validation),
                     callbacks=callbacks,
                     )  #,validation_split = 0.2

## show train_accuracy and val_accuracy in plot
accroucy = epoch_hist.history['accuracy']
val_accuracy = epoch_hist.history['val_accuracy']
loss = epoch_hist.history['loss']
val_loss = epoch_hist.history['val_loss']
epochs= range(len(accroucy))
plt.plot(epochs,accroucy,'-bo',label='Trainin Accroucy')
plt.plot(epochs,val_accuracy,'-rs',label='valid Accroucy')
plt.legend()
plt.show()

## eval test data
evaluation    = Cnn.evaluate(X_test,y_test)
print("="*100)
print("Test Accroucy is: ",evaluation[1])
## save model plot
keras.utils.plot_model(Cnn, "CNN.png", show_shapes=True)
