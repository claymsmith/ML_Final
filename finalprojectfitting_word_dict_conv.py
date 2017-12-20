import pickle
import keras
import numpy as np
from sklearn.utils import resample
from sklearn.utils import shuffle
from keras.models import Model, Sequential, optimizers
from keras.layers import Dense, Activation, LSTM, Dropout, Conv1D, Flatten, TimeDistributed, Embedding, GlobalMaxPooling1D

X, y, vsize = pickle.load(open('X_y_word_dict.p', 'rb'))

#basic data parameters
nSamp = X.shape[0]
nBait = np.sum(y)
nReal = nSamp - nBait

#upsample minority classes in X
x_maj = X[y==0, :]
x_min = X[y==1, :]
x_min_upsample = resample(x_min, n_samples = nReal)
XFull = np.concatenate((x_maj, x_min_upsample), axis = 0)
yFull = np.concatenate((np.zeros(nReal), np.ones(nReal)))

#and now shuffle the upsampled data in the same way:
XFull, yFull = shuffle(XFull, yFull, random_state = 123)

#training and test split, might do something with this for kfold
ntr = 15000
Xtr = XFull[:ntr, :]
Xts = XFull[ntr:, :]
ytr = yFull[:ntr]
yts = yFull[ntr:]

Xtr3D = np.expand_dims(Xtr, axis=2)
Xts3D = np.expand_dims(Xts, axis=2)

#here goes our keras model
model = Sequential()
model.add(Conv1D(512,kernel_size = 16, input_shape = (Xtr3D.shape[1:]) , activation='relu', name = 'C1'))
model.add(Dropout(0.5))
model.add(Conv1D(256,kernel_size = 16, activation='relu', name = 'C2'))
#model.add(Conv1D(128,kernel_size = 16, activation='relu', name = 'C3'))
model.add(Flatten())
#model.add(Dense(128))
model.add(Dropout(0.5)) #to combat a bit of overfitting
#model.add(Activation('sigmoid'))
model.add(Dense(1, activation = 'sigmoid', name = 'output'))

#callback exactly like we did on our lab
class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.loss = []
		self.val_acc = []
	def on_batch_end(self, batch, logs={}):
		self.loss.append(logs.get('loss'))
	def on_epoch_end(self, epoch, logs):
		self.val_acc.append(logs.get('val_acc'))

history_cb = LossHistory()
opt = optimizers.Adam(lr = 0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 250
nEpoch = 250
model.fit(Xtr3D, ytr, callbacks = [history_cb], verbose = 1, epochs=nEpoch, batch_size=batch_size, validation_data=(Xts3D,yts))

#save model and loss and acc values
model.save('Model_CNN_w_dropout.h5')
losses = history_cb.loss
valaccs = history_cb.val_acc
pickle.dump( [losses, valaccs], open("Loss_Val_acc_CNN_w_dropout.p", "wb"))
