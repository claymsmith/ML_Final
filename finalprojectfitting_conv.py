import pickle
import keras
import numpy as np
from keras.models import Model, Sequential, optimizers
from keras.layers import Dense, Activation, LSTM, Dropout, Conv1D, Flatten, TimeDistributed, Embedding, GlobalMaxPooling1D

keras.backend.clear_session()
Xtr, ytr, vsize = pickle.load(open( "Xtr_ytr_full.p", "rb" ))
Xts, yts = pickle.load(open( "Xts_yts_full.p", "rb" ))
ntr = 15000
nts = Xts.shape[0]-ntr
nfeat = Xts.shape[1]
Xts = (Xts*2)/(vsize-1)-1
Xtr1 = Xts[:ntr, :]
ytr1 = yts[:ntr] #truncate some of our validation
Xts1 = Xts[ntr:, :]
yts1 = yts[ntr:]

seqL = Xtr.shape[1]
nStr = int(np.ceil(ntr/vsize))
nSts = int(np.ceil(nts/vsize))

Xtr3D = np.expand_dims(Xtr1, axis=2)
Xts3D = np.expand_dims(Xts1, axis=2)

model = Sequential()
model.add(Conv1D(128,kernel_size = 8, input_shape = (Xtr3D.shape[1:]) , activation='sigmoid'))
model.add(Conv1D(64,kernel_size = 8, activation='sigmoid'))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('sigmoid'))
model.add(Dense(1, activation = 'sigmoid', name = 'output'))


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

opt = optimizers.adam(lr = 0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 500
model.fit(Xtr3D, ytr1, callbacks = [history_cb], verbose = 1, epochs=240, batch_size=batch_size, validation_data=(Xts3D,yts1))

#save model and loss and acc values
model.save('Model_CNN.h5')
losses = history_cb.loss
valaccs = history_cb.val_acc
pickle.dump( [losses, valaccs], open("Loss_Val_acc_CNN.p", "wb"))