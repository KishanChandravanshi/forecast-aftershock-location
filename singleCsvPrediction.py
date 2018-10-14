import numpy as np
import helper_functions
from collections import defaultdict
import sklearn

################################## INITIALISATION #########################
# Change the filename according to your needs
filename = 'Data/singleCSV/1978MIYAGI01YAMA_grid.csv'
weightFile = 'Data/weights.h5'
# predFile contains the X and y, that we obtained when we processed the csv
predFile = 'Data/singleCSV/singlePred.h5'
columns = ['stresses_full_xx', 'stresses_full_yy', 
           'stresses_full_xy', 'stresses_full_xz',
           'stresses_full_yz','stresses_full_zz']
# testFile basically is the file in which we have saved the data in an organised way
# it is this that will be be used to create X and y to be fed to the model
testFile = 'single.h5'

################################ BUILDING DATA ##########################
# first we need to convert the csv into dictionary and
# then we will save it to a .h5 file naming 'single.h5'
dictionary = defaultdict(list)
print('working with {}, please wait...'.format(filename.split('/')[-1]))
data = helper_functions.csv2dict(filename)
grid_aftershock_count = np.double(data['aftershocksyn'])
temp = grid_aftershock_count.tolist()
dictionary['aftershocksyn'].extend(temp)
for column in columns:
    dictionary[column].extend(np.double(data[column]))


columns.append('aftershocksyn')

# now we have all our information in dictionary format, let's save it in a single.h5
helper_functions.dict2HDF('single.h5', columns, dictionary)

########################### Predicting part ################################
# name of features in our dataset
features_in = ['stresses_full_xx',
               'stresses_full_yy',
               'stresses_full_xy',
               'stresses_full_xz',
               'stresses_full_yz',
               'stresses_full_zz']

# name of label
features_out = 'aftershocksyn'
# load the model
model = helper_functions.createModel()
# load the weights and evaluate on test file
model.load_weights(weightFile)
X, y = helper_functions.loadDataFromHDF(testFile, features_in, features_out)
y_pred = model.predict(X)
helper_functions.writeHDF(predFile, X, y)
auc = sklearn.metrics.roc_auc_score(y, y_pred)
