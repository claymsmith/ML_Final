import matplotlib.pyplot as plt
import pickle
from keras.models import Model, load_model
import numpy as np

Model_no_dropout = load_model('Model_CNN_no_dropout.h5')
Model_dropout = load_model('Model_CNN_w_dropout.h5')
print(Model_no_dropout.summary())
print(Model_dropout.summary())
loss_n, val_acc_n = pickle.load(open('Loss_Val_acc_CNN_no_dropout.p', 'rb'))
loss, val_acc = pickle.load(open('Loss_Val_acc_CNN_w_dropout.p', 'rb'))

Epochs = np.linspace(1, 250, len(val_acc_n))
batch_size = 250
batches = np.linspace(1, 250, len(loss))

plt.subplot(1, 2, 1)
plt.plot(batches, loss_n)
plt.title('Training Loss vs. Epoch')
plt.subplot(1, 2, 2)
plt.plot(Epochs, val_acc_n)
plt.title('Validation Accuracy vs. Epoch')
plt.show()

plt.subplot(1, 2, 1)
plt.plot(batches, loss)
plt.title('Training Loss vs. Epoch')
plt.subplot(1, 2, 2)
plt.plot(Epochs, val_acc)
plt.title('Validation Accuracy vs. Epoch')
plt.show()
