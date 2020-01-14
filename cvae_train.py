'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from keras.layers import Lambda, Input, Dense, LSTM, RepeatVector, Conv2D, Flatten, Conv2DTranspose, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K, objectives
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import librosa.display

import pretty_midi
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping

import librosa
from scipy.io import wavfile as wav




intermediate_dim = 128
batch_size = 1
latent_dim = 2
epochs = 100
random_state = 42
dataset_size = 10000
list_files_name= []
file_shuffle=[]
test_size=0.25
filters = 16
kernel_size = 2
size_mfcc = 40
min_size_file = 2147483646

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data



    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    #plt.savefig(filename)
    #print(len(z_mean))
    nb_elem_per_class = dataset_size*test_size

    #to_decode = np.array([[0.5, 0], [1.8, 1]], dtype=np.float32)
    #final = decoder.predict(to_decode)
    #print(final )

    annotate=False
    if annotate:
        for i, txt in enumerate(file_shuffle[ :int(dataset_size*2 * test_size)]):
            txt = txt.replace('.mid', '')
            txt =txt.replace('enerated', '')
            txt =txt.replace('andom', '')
            txt = txt.replace('_', '')
            plt.annotate(txt,(z_mean[i,0], z_mean[i,1]))

    plt.show()


def load_data(path, class_label, index_filename, min_size_file ):

    path, dirs, files = next(os.walk(path))
    num_size = len(dirs)
    current_folder = 0
    num_files = 0

    for subdir, dirs, files in os.walk(path):
        for file in files:
            if num_files < dataset_size:
                if file != ".DS_Store":
                    file_path = path + file
                    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
                    if len(audio)<min_size_file:
                        min_size_file=len(audio)

    print("taille du plus petit fichier audio: ", min_size_file)




    for subdir, dirs, files in os.walk(path):
        for file in files:
            if num_files < dataset_size:
                if file != ".DS_Store":
                    # print(os.path.join(subdir, file))
                    #try:
                    file_path= path+file
                    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
                    audio = audio[:min_size_file]

                    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                    #print(mfccs)
                    #pad_width = max_pad_len - mfccs.shape[1]
                    #print(pad_width)
                    #mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
                    print(mfccs.shape)
                    features.append([mfccs, num_files])
                    #except Exception as e:
                    #    print("Error encountered while parsing file: ", file)
                    #    return None

                    #return mfccsscaled
                    list_files_name.insert(index_filename+num_files, file)
                    num_files += 1

        current_folder += 1
        print("Done ", num_files, " from ", current_folder, " folders on ", num_size)
        return min_size_file


print("LOADING DATA FOR TRAINING...")
features = []

#path_to_load = "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/0"
#load_data(path_to_load, 0,   0)
#path_to_load = "/home/kyrillos/CODE/VAEMIDI/quantized_rythm_dataset_v2_temperature/100"
path_to_load = "/Users/Cyril_Musique/Documents/Cours/M2/WAVAE_audio_vae/audio_files/"

min_size_file= load_data(path_to_load,1,  dataset_size, min_size_file)

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

# Convert features & labels into numpy arrays
listed_feature = featuresdf.feature.tolist()

X = np.array(listed_feature).astype(object)
y = np.array(featuresdf.class_label.tolist())

print(X.shape, y.shape)
# split the dataset


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
X_shuffle = shuffle(X, random_state=random_state)
y_shuffle = shuffle(y, random_state=random_state)
file_shuffle = shuffle(list_files_name, random_state=random_state)

print(min_size_file)
x_train = np.reshape(x_train, [x_train.shape[0], size_mfcc, X.shape[2], 1])
x_test = np.reshape(x_test, [x_test.shape[0],size_mfcc, X.shape[2], 1])



#x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float64') / 10000
x_test = x_test.astype('float64') / 10000



#Convolutional VAE

#ENCODER
input_shape = (size_mfcc, X.shape[2], 1) #datasize
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()


#DECODER
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# use Conv2DTranspose to reverse the conv layers from the encoder
for i in range(2):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

decoder.summary()


#Building the VAE
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

#LOSS
use_mse=True
if use_mse:
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
else:
    reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                              K.flatten(outputs))

reconstruction_loss *= size_mfcc * (min_size_file)
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)


#Compile the VAE
vae.compile(optimizer='rmsprop')
vae.summary()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)


    if args.weights:
        print("LOADING WEIGHTS")
        vae.load_weights(args.weights)
    else:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        # train the autoencoder

        score=vae.fit(x_train,
                epochs=epochs,
                verbose=1,
                batch_size=batch_size,
                validation_data=(x_test, None),
                callbacks=[es])
        vae.save_weights('vae_mlp_mnist.h5')

        score2 = vae.evaluate(x_test, None, verbose=1)
        print('Score', score.history)
        print('Score', score2)




    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")

