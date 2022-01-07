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

# Preprocess the data and normalization
X_train = X_train.reshape(60000, 784).astype("float32") / 255
X_test = X_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

## use shuffle for best performance in train
X_train, y_train = shuffle(X_train, y_train)

## sperat validation dataset from train data
ValidationPersent = 20
N_trainData = len(X_train)
N_validData = int((N_trainData*ValidationPersent)/100)
x_validation,y_validation = X_train[0:N_validData],y_train[0:N_validData]
## remove validation data from training data
X_train = X_train[N_validData+1:]
y_train = y_train[N_validData+1:]


## create best optimizer
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

## Let's build the CNN

DNN = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(784,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
DNN.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
DNN.summary()


callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=2,
        verbose=1,
    )
]

epoch_hist = DNN.fit(X_train,y_train,
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
evaluation    = DNN.evaluate(X_test,y_test)
print("="*100)
print("Test Accroucy is: ",evaluation[1])
## save model plot
keras.utils.plot_model(DNN, "DNN.png", show_shapes=True)
