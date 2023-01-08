from keras.models import model_from_json
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D  # , AveragePooling1D
from keras.layers import Flatten, Dropout, Activation  # Input,
from keras.layers import Dense  # , Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from utils import dataset
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

"""### Setup the Basic Paramter"""
dataset_path = os.path.abspath('./Dataset')
destination_path = os.path.abspath('./')
randomize = True  # To shuffle the dataset instances/records
split = 0.8  # for splitting dataset into training and testing dataset
sampling_rate = 20000  # Number of sample per second e.g. 16KHz
emotions = ["positive", "negative", "neutral"]

df, train_df, test_df = dataset.create_and_load_meta_csv_df(dataset_path, destination_path, randomize, split)
print('Dataset samples  : ', len(df), "\nTraining Samples : ", len(train_df), "\ntesting Samples  : ", len(test_df))

"""
### Labels Assigned for emotions : 
- 0 : positive
- 1 : negative
- 2 : neutral
"""

"""#Data Pre-Processing
### Getting the features of audio files using librosa
Calculating MFCC, Pitch, magnitude, Chroma features.
"""
# (added)
# with xvector
#
from utils.feature_extraction import get_features_dataframe
# trainfeatures_noised_plusXvector, trainlabel_noised_plusXvector = get_features_dataframe(train_df, sampling_rate,
#                                                                                          add_noise=True)
# trainfeatures_noised_plusXvector.to_pickle('./features_dataframe/trainfeatures_noised_plusXvector')
# trainlabel_noised_plusXvector.to_pickle('./features_dataframe/trainlabel_noised_plusXvector')
#
# trainfeatures_plusXvector, trainlabel_plusXvector = get_features_dataframe(train_df, sampling_rate)
# trainfeatures_plusXvector.to_pickle('./features_dataframe/trainfeatures_plusXvector')
# trainlabel_plusXvector.to_pickle('./features_dataframe/trainlabel_plusXvector')
#
# testfeatures_plusXvector, testlabel_plusXvector = get_features_dataframe(test_df, sampling_rate)
# testfeatures_plusXvector.to_pickle('./features_dataframe/testfeatures_plusXvector')
# testlabel_plusXvector.to_pickle('./features_dataframe/testlabel_plusXvector')

# I have run above 2 lines and get the featured dataframe.
# and store it into pickle file to use it for later purpose.
# it takes too much time to generate features(around 30-40 minutes).


trainfeatures_noised = pd.read_pickle('./features_dataframe/trainfeatures_noised_plusXvector')
trainlabel_noised = pd.read_pickle('./features_dataframe/trainlabel_noised_plusXvector')

trainfeatures_original = pd.read_pickle('./features_dataframe/trainfeatures_plusXvector')
trainlabel_original = pd.read_pickle('./features_dataframe/trainlabel_plusXvector')

trainfeatures = trainfeatures_noised.append(trainfeatures_original, ignore_index=True)
trainlabel = trainlabel_noised.append(trainlabel_original, ignore_index=True)

testfeatures = pd.read_pickle('./features_dataframe/testfeatures_plusXvector')
testlabel = pd.read_pickle('./features_dataframe/testlabel_plusXvector')

trainfeatures = trainfeatures.fillna(0)
testfeatures = testfeatures.fillna(0)
# (\added)

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel).ravel()
X_test = np.array(testfeatures)
y_test = np.array(testlabel).ravel()

# One-Hot Encoding
lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

"""### Changing dimension for CNN model"""

x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)

"""# 6. Model Creation"""

model = Sequential()
model.add(Conv1D(256, 5, padding='same',
                 input_shape=(x_traincnn.shape[1], x_traincnn.shape[2])))
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5, padding='same', ))
model.add(Activation('relu'))
model.add(Conv1D(128, 5, padding='same', ))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train:
cnnhistory = model.fit(x_traincnn, y_train, batch_size=16, epochs=50, validation_data=(x_testcnn, y_test))

# Save model and weights
model_name = 'model_noise_xvector.h5'
save_dir = os.path.join(os.getcwd(), 'Trained_Models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print('Saved trained model at %s ' % model_path)

"""### Loading the model"""

# loading json and creating model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./Trained_Models/model_noise_xvector.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
