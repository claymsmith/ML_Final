import pickle
import keras
import numpy as np
from keras.models import Model, Sequential, optimizers
from keras.layers import Dense, Activation, LSTM, Dropout, Conv1D

keras.backend.clear_session()

#load our data from the full 
Xtr, ytr, vsize = pickle.load(open( "Xtr_ytr_full.p", "rb" ))
Xts, yts = pickle.load(open( "Xts_yts_full.p", "rb" ))

Xtr1 = Xts[:15000, :1000]/vsize
ytr1 = yts[:15000] #truncate some of our validation
Xts1 = Xts[15000:, :1000]/vsize
yts1 = yts[15000:]



#print(np.sum(yts))
nin = Xtr1.shape[1]
#getting our model shape parameters
model = Sequential()
model.add(Dense(64, input_shape = (nin,), activation = 'sigmoid', name = 'hidden1'))
model.add(Dropout(0.5))
#model.add(Dense(16, activation = 'sigmoid', name = 'hidden3'))
#model.add(Dropout(0.5))
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

history_cb = LossHistory()

opt = optimizers.adam(lr = .01)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 250
model.fit(Xtr1, ytr1, callbacks = [history_cb], verbose = 1, epochs=10000, batch_size=batch_size, validation_data=(Xts1,yts1))

#works fairly well, still a bit overfit... trying conv layers
