import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

x_train = np.array(train.iloc[:, 1:])
y_train = to_categorical(np.array(train.iloc[:, 0]))

x_test = np.array(test.iloc[:, 1:])
ids = np.array(test.iloc[:, 0])

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

batch_size = 256
num_classes = 10
epochs = 200
dropout = 0.37

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', padding = 'same', kernel_initializer='he_normal', input_shape=input_shape))
model.add(Conv2D(16, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(dropout))

model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(dropout))

model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(dropout))

model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(dropout))

model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(dropout))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(decay=0.001),
              metrics=['accuracy'])

checkpoint_callback = keras.callbacks.ModelCheckpoint(directory+ "model_cnn_v6_{epoch:03d}.hdf5",
                                                      monitor='val_loss', 
                                                      verbose=0, 
                                                      save_best_only=False, 
                                                      save_weights_only=False, 
                                                      mode='auto', 
                                                      period=10)

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=1/10,
          callbacks = [checkpoint_callback],
          shuffle=True)

%matplotlib inline
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b-', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b-', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

preds = model.predict_classes(x_test).reshape((10000, 1))
ids = ids.reshape((10000, 1))
output = np.hstack((ids, preds))
output = pd.DataFrame(data = output, columns = ["ids", "label"])
output.to_csv(path_or_buf = directory + "preds.csv", index = False, line_terminator = "\r\n")