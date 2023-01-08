from keras.models import model_from_json
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D  # , AveragePooling1D
from keras.layers import Flatten, Dropout, Activation  # Input,
from keras.layers import Dense  # , Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from utils import dataset
import pandas as pd
import warnings
import random
from matplotlib import pyplot as plt
import time

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
# from utils.feature_extraction import get_features_dataframe
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
    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax'))
    opt = keras.optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
score_list = []
time_train = []
time_eval = []
scores = []
scaler = StandardScaler()
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

print(f'Variance described by the first 5 principle components is {sum(pca.explained_variance_ratio_[:4])}')
    # test = pca.explained_variance_ratio_[:30]
    # print(test)
    # 0.81% variance of the data is good to go, now we will trian the model on PCA(n_components=5)

x_traincnn_filtered = x_traincnn[:, :200]
x_testcnn_filtered = x_testcnn[:, :200]

    # Changing dimension for CNN model
x_traincnn_filtered = np.expand_dims(x_traincnn_filtered, axis=2)
x_testcnn_filtered = np.expand_dims(x_testcnn_filtered, axis=2)
model = create_model(x_traincnn_filtered)
start_train = time.time()  # time
cnnhistory = model.fit(x_traincnn_filtered, y_train, batch_size=32, epochs=50,
                           validation_data=(x_testcnn_filtered, y_test))
end_train = time.time()  # time
#     # append score
start_eval = time.time()  # time
start_eval = time.time()

score = model.evaluate(x_testcnn_filtered, y_test)
print(f'Score on {x_traincnn_filtered.shape[1]} features is {score[1]}')
scores.append(score[1])
print(scores)
end_eval = time.time()  # time
score_list.append(score[1])
time_eval.append(end_eval - start_eval)  # time
time_train.append(end_train - start_train)  # time
#
df = pd.DataFrame({'pca components': 34, 'accuracy on test': score_list, 'train_time': time_train,
                   'eval_time': time_eval})
print(df.to_string(index=False))

# # tuning pca - with component_number_search
# component_number = np.linspace(10, X_train.shape[1], 5).astype(int)
# score_list = []
# time_train = []
# time_eval = []
# for c in component_number:
#     pca = PCA(n_components=c)
#     x_traincnn = pca.fit_transform(X_train)
#     x_testcnn = pca.transform(X_test)
#
#     # Changing dimension for CNN model
#     x_traincnn = np.expand_dims(x_traincnn, axis=2)
#     x_testcnn = np.expand_dims(x_testcnn, axis=2)
#     model = create_model(x_traincnn)
#     print(f'training on {x_traincnn.shape[1]} pca picked features')
#     start_train = time.time()  # time
#     cnnhistory = model.fit(x_traincnn, y_train, batch_size=32, epochs=5, validation_data=(x_testcnn, y_test))
#     end_train = time.time()  # time
#     # append score
#     start_eval = time.time()  # time
#     score = model.evaluate(x_testcnn, y_test)
#     end_eval = time.time()  # time
#     score_list.append(score[1])
#     time_eval.append(end_eval - start_eval)  # time
#     time_train.append(end_train - start_train)  # time
#
# df = pd.DataFrame({'pca components': component_number, 'accuracy on test': score_list, 'train_time': time_train,
#                    'eval_time': time_eval})
# print(df.to_string(index=False))

# (/added)
