from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils.feature_extraction import *
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
from sklearn.decomposition import PCA
from utils import dataset
import pandas as pd
import warnings
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')

"""reproduce results"""
np.random.seed(0)
keras.utils.set_random_seed(0)
random.seed(0)

"""### Setup the Basic Paramter"""
dataset_path = os.path.abspath('./Dataset')
destination_path = os.path.abspath('./')
randomize = True  # To shuffle the dataset instances/records
split = 0.8  # for splitting dataset into training and testing dataset
sampling_rate = 20000  # Number of sample per second
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

# I have run above 12 lines and get the featured dataframe.
# and store it into pickle file to use it for later purpose.
# it takes too much time to generate features(around 30-40 minutes).


trainfeatures_noised = pd.read_pickle('./features_dataframe/trainfeatures_noised_plusXvector')
trainlabel_noised = pd.read_pickle('./features_dataframe/trainlabel_noised_plusXvector')

trainfeatures_original = pd.read_pickle('./features_dataframe/trainfeatures_plusXvector')
trainlabel_original = pd.read_pickle('./features_dataframe/trainlabel_plusXvector')

# append augmentation
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

"""# 6. Model Creation"""


# (added)
def create_model(x_traincnn):
    # x_traincnn only used for understanding the dimension of the input layer
    model = Sequential()
    model.add(Conv1D(256, 5, padding='same',
                     input_shape=(x_traincnn.shape[1], x_traincnn.shape[2])))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1]))  # class number
    model.add(Activation('softmax'))
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=577)
x_traincnn = pca.fit_transform(X_train)
x_testcnn = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# plot scree graph:
plt.plot(pca.explained_variance_ratio_[:20], 'o-', linewidth=2,
         color='blue')  # showing all the 577 features would be caos so ill show only 20principal components
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.savefig('Scree_plot_pca')
plt.show()

elbow_num = 30

print(f'Variance described by the first {elbow_num} principle components is {sum(pca.explained_variance_ratio_[:100])}')
# 0.59% variance of the data is good to go, now we will train the model on PCA(n_components=7)

x_traincnn_filtered = x_traincnn[:, :elbow_num]
x_testcnn_filtered = x_testcnn[:, :elbow_num]

# Changing dimension for CNN model
x_traincnn_filtered = np.expand_dims(x_traincnn_filtered, axis=2)
x_testcnn_filtered = np.expand_dims(x_testcnn_filtered, axis=2)
model = create_model(x_traincnn_filtered)
cnnhistory = model.fit(x_traincnn_filtered, y_train, batch_size=32, epochs=50,
                       validation_data=(x_testcnn_filtered, y_test))
score = model.evaluate(x_testcnn_filtered, y_test)
print(f'Score on {x_traincnn_filtered.shape[1]} features is {score[1]}')


# confusion matrix with pca:
y_test_pred_probs = model.predict(x_testcnn_filtered, batch_size=32, verbose=1)
y_test_pred = np.argmax(y_test_pred_probs, axis=1)
y_test = np.argmax(y_test, axis=1)
cf_matrix = confusion_matrix(y_test, y_test_pred)

ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(emotions)
ax.yaxis.set_ticklabels(emotions)
plt.show()


# predicting folder
predict_folder_images(classifier=model, sampling_rate=sampling_rate, folder_name='new_sounds', pca=pca , elbow_num=elbow_num)
