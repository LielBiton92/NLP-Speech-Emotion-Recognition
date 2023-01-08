# feature_extracting
import glob

import librosa
import pandas as pd
import numpy as np
# from tqdm import tqdm
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
from tqdm import tqdm


# augment data by adding noise(added)
def gen_noise_data(data):
    Nsamples = len(data)
    noiseSigma = 0.01
    noiseAmplitude = 0.5
    noise = noiseAmplitude * np.random.normal(0, noiseSigma, Nsamples)
    noised_data = noise + data
    return noised_data, noise


# (\added)


def get_audio_features(audio_path, sampling_rate, classifier=None, add_noise=False):
    X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=sampling_rate * 2, offset=0.5)

    # augmented data by added noise(added)
    if add_noise:
        X, noise = gen_noise_data(X)
    # (\added)

    sample_rate = np.array(sample_rate)
    y_harmonic, y_percussive = librosa.effects.hpss(X)
    pitches, magnitudes = librosa.core.pitch.piptrack(y=X, sr=sample_rate)

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=1)

    pitches = np.trim_zeros(np.mean(pitches, axis=1))[:20]

    magnitudes = np.trim_zeros(np.mean(magnitudes, axis=1))[:20]

    C = np.mean(librosa.feature.chroma_cqt(y=y_harmonic, sr=sampling_rate), axis=1)

    # xvector(added)
    if classifier == None:
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                    savedir="pretrained_models/spkrec-xvect-voxceleb")

    speechbrain_X = torch.tensor([X])
    xvector = np.squeeze(classifier.encode_batch(speechbrain_X))
    if len(xvector.shape) > 1:
        # there is rare files that generate duplicate tensors (2,512) the code below made to overcome dimension problem
        xvector = xvector[0]
        print(audio_path)
    xvector = np.array(xvector)
    # (\added)
    return [mfccs, pitches, magnitudes, C, xvector]


def get_features_dataframe(dataframe, sampling_rate, add_noise=False):
    labels = pd.DataFrame(dataframe['label'])
    # spitchbrain classifier(added)
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                savedir="pretrained_models/spkrec-xvect-voxceleb")
    # (\added)
    features = pd.DataFrame(columns=['mfcc', 'pitches', 'magnitudes', 'C', 'xvector'])
    for index, audio_path in tqdm(enumerate(dataframe['path'])):
        features.loc[index] = get_audio_features(audio_path, sampling_rate, classifier, add_noise)

    mfcc = features.mfcc.apply(pd.Series)
    pit = features.pitches.apply(pd.Series)
    mag = features.magnitudes.apply(pd.Series)
    C = features.C.apply(pd.Series)
    xvector = features.xvector.apply(pd.Series)

    combined_features = pd.concat([mfcc, pit, mag, C, xvector], axis=1, ignore_index=True)

    return combined_features, labels



def get_features_exact(paths, sampling_rate, add_noise=False):
    # spitchbrain classifier(added)
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                savedir="pretrained_models/spkrec-xvect-voxceleb")
    # (\added)
    features = pd.DataFrame(columns=['mfcc', 'pitches', 'magnitudes', 'C', 'xvector'])
    for index, audio_path in tqdm(enumerate(paths)):
        features.loc[index] = get_audio_features(audio_path, sampling_rate, classifier, add_noise)

    mfcc = features.mfcc.apply(pd.Series)
    pit = features.pitches.apply(pd.Series)
    mag = features.magnitudes.apply(pd.Series)
    C = features.C.apply(pd.Series)
    xvector = features.xvector.apply(pd.Series)

    combined_features = pd.concat([mfcc, pit, mag, C, xvector], axis=1, ignore_index=True)

    return combined_features

def predict_folder_images(classifier, sampling_rate, folder_name='new_sounds', pca=None , elbow_num=10):
    wav_locs = []
    emotions = ["positive", "negative", "neutral"]

    for file in glob.glob(f'./{folder_name}/*.wav'):
        # print(file)
        wav_locs.append(file)

    features = get_features_exact(wav_locs, sampling_rate, add_noise=False).fillna(0)


    if pca:
        features = pca.transform(features)
        features = features[:, :elbow_num]

    features = np.expand_dims(features, axis=2)

    y_pred = classifier.predict(features, batch_size=32, verbose=1)
    y_pred = np.argmax(y_pred,axis=1)
    for audio,pred in list(zip(wav_locs,y_pred)):
        print(f'{audio} was predicted as {emotions[pred]}')

#
# if __name__ == "__main__":
#     demo_audio_path = './demo_audio.wav'
#     get_audio_features(demo_audio_path, 20000)
