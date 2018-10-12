import helper_functions
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import copy
import numpy as np
# name of weights file and training/ testing files
weightFile = 'Data/weights.h5'
trainFile = 'Data/training.h5'
testFile = 'Data/testing.h5'

# name of the features that we are dealing with
# features_in -> on which our model will train
# features_out -> which our model will predict
features_in = ['stresses_full_xx', 'stresses_full_yy', 'stresses_full_xy', 'stresses_full_xz','stresses_full_yz','stresses_full_zz']
features_out = 'aftershocksyn'

# load the training dataset
X, y = helper_functions.loadDataFromHDF(trainFile, features_in, features_out)

# Let's create a validation set, with positive grid cells and negative grid
# so that validation contain equal number of elements of both classes
# let's look for positive indexes
posIndex = np.where(y == 1) # it will return a list having the index of row with y == 1
posSize = np.size(posIndex) # total number of the elements having y==1
negIndex = np.where(y == 0)
negSize = np.size(negIndex)

# now divide the data into positive and negative samples
posData = np.column_stack((np.squeeze(X[posIndex, :]), y[posIndex].T))
negData = np.column_stack((np.squeeze(X[negIndex, :]), y[negIndex].T))

# let's shuffle the dataset
np.random.seed(5)
np.random.shuffle(posData)
np.random.shuffle(negData)
np.random.seed()

######################## CREATION OF VALIDATION SET ####################
# we will create a validation dataset consisting of 10% of positive samples and negative samples
cutoff = int(round(posSize / 10)) # 10% of the total number

# creating a positive validation samples
Xp_val = copy.copy(posData[:cutoff, :len(features_in) * 2])
yp_val = copy.copy(posData[:cutoff, len(features_in) * 2])

# creating a negative validataion samples
Xn_val = copy.copy(negData[:cutoff, :len(features_in) * 2])
yn_val = copy.copy(negData[:cutoff, len(features_in) * 2])

# Now we need to merge these two validation dataset
X_val = np.row_stack((Xp_val, Xn_val))
y_val = np.append(yp_val, yn_val)

############################## TRAINING SET #####################################
# the remaining dataset is our training set
posDataFinal = copy.copy(posData[cutoff:, :])
negDataFinal = copy.copy(negData[cutoff:, :])

shapeOfTrainingDataset = np.shape(posDataFinal)

############################## HYPERPARAMETER ###############################
# set hyperparameters
batch_size = 3500
steps_per_epoch = int(round((shapeOfTrainingDataset[0]) / batch_size))
epochs = 100
posStart = 0
negStart = 0

#################################### TRAIN #################################
model = helper_functions.createModel()
checkpointer = ModelCheckpoint(filepath=weightFile, monitor='val_loss', verbose=2, save_best_only=True)
history = model.fit_generator(helper_functions.generateData(posDataFinal, negDataFinal, batch_size, posStart, negStart), steps_per_epoch, validation_data=(X_val, y_val), callbacks=[checkpointer], verbose=2, epochs=epochs)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
