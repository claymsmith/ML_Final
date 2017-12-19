import pickle
import keras
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers

keras.backend.clear_session()

#load our data from the full 
Xtr, ytr = pickle.load(open("Xtr_ytr_MnC.p", "rb"))
Xts, yts = pickle.load(open("Xts_yts_MnC.p", "rb"))

nh = 512
nin = Xtr.shape[1]

model = Sequential()
model.add(Dense(nh, input_shape = (nin,), activation = 'sigmoid', name = 'hidden'))
model.add(Dense(1, activation='sigmoid', name = 'output'))
print(model.summary())

class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.loss = []
		self.val_acc = []
	def on_batch_end(self, batch, logs={}):
		self.loss.append(logs.get('loss'))
	def on_epoch_end(self, epoch, logs):
		self.val_acc.append(logs.get('val_acc'))

# Create an instance of the history callback
history_cb = LossHistory()

opt = optimizers.Adam(lr=.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 10

model.fit(Xtr, ytr, callbacks = [history_cb], verbose = 1, epochs=100, batch_size=batch_size, validation_data=(Xts,yts))

