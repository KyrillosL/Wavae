
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import glob
import time
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
from multiprocessing import Pool, Value

from pathlib import Path

import concurrent.futures

import multiprocessing as mp

import pretty_midi
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping

import librosa
from scipy.io import wavfile as wav
from pydub import AudioSegment

class CVae:
    def __init__(self):

        self.file_to_write_X = "/home/kyrillos/CODE/Wavae/audios/data_X"
        self.file_to_write_y = "/home/kyrillos/CODE/Wavae/audios/data_y"
        self.path_to_load = "/home/kyrillos/CODE/Wavae/audios/amusique/"

        self.intermediate_dim = 128
        self.batch_size = 1
        self.latent_dim = 2
        self.epochs = 100
        self.random_state = 42
        self.dataset_size = 10000
        self.list_files_name= []
        self.file_shuffle=[]
        self.test_size=0.25
        self.filters = 16
        self.kernel_size = 2
        self.size_mfcc = 40
        self.min_size_file = 2147483646

        self.smallest_file = self.get_smallest_file()


    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


    def plot_results(self):

        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = self.encoder.predict(self.x_test)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=self.y_test)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")

        annotate=False
        if annotate:
            for i, txt in enumerate(self.file_shuffle[ :int(self.dataset_size*2 * self.test_size)]):
                txt = txt.replace('.mid', '')
                txt =txt.replace('enerated', '')
                txt =txt.replace('andom', '')
                txt = txt.replace('_', '')
                plt.annotate(txt,(z_mean[i,0], z_mean[i,1]))

        plt.show()


    def load_data(self):
        self.features = []
        print("LOADING DATA FOR TRAINING...")
        path, dirs, files = next(os.walk(self.path_to_load))
        num_size = len(dirs)
        current_folder = 0
        num_files = 0

        audiomin, sample_ratemin = librosa.load(self.smallest_file, res_type='kaiser_fast')

        self.min_size_file=len(audiomin)



        print("taille du plus petit fichier audio: ", self.min_size_file)

        for subdir, dirs, files in os.walk(path):
            for file in files:
                if num_files < self.dataset_size:
                    if file != ".DS_Store":
                        # print(os.path.join(subdir, file))
                        try:
                            file_path= subdir+"/"+file
                            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
                            #audio = audio[:self.min_size_file]
                            #audio = audio[:2827]

                            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

                            print(mfccs.shape)
                            self.features.append([mfccs, num_files])
                        except Exception as e:
                            print("Error encountered while parsing file: ", file)
                            return None

                        self.list_files_name.insert(num_files, file)
                        num_files += 1

            current_folder += 1
            print("Done ", num_files, " from ", current_folder, " folders on ", num_size)


    def process_data(self):
        featuresdf = pd.DataFrame(self.features, columns=['feature', 'class_label'])

        print('Finished feature extraction from ', len(featuresdf), ' files')

        # Convert features & labels into numpy arrays
        listed_feature = featuresdf.feature.tolist()

        self.X = np.array(listed_feature).astype(object)
        self.y = np.array(featuresdf.class_label.tolist())

        print(self.X.shape, self.y.shape)

    def split_data(self):

        self.x_train, self.x_test,self. y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        X_shuffle = shuffle(self.X, random_state=self.random_state)
        y_shuffle = shuffle(self.y, random_state=self.random_state)
        self.file_shuffle = shuffle(self.list_files_name, random_state=self.random_state)

        print(self.min_size_file)
        self.x_train = np.reshape(self.x_train, [self.x_train.shape[0], self.size_mfcc, self.X.shape[2], 1])
        self.x_test = np.reshape(self.x_test, [self.x_test.shape[0],self.size_mfcc, self.X.shape[2], 1])

        self.x_train = self.x_train.astype('float64') / 10000
        self.x_test = self.x_test.astype('float64') / 10000



    def compile(self):

        #ENCODER
        input_shape = (self.size_mfcc, self.X.shape[2], 1) #datasize
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        for i in range(2):
            self.filters *= 2
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)

        # shape info needed to build decoder model
        shape = K.int_shape(x)

        # generate latent vector Q(z|X)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()


        #DECODER
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        # use Conv2DTranspose to reverse the conv layers from the encoder
        for i in range(2):
            x = Conv2DTranspose(filters=self.filters,
                                kernel_size=self.kernel_size,
                                activation='relu',
                                strides=2,
                                padding='same')(x)
            self.filters //= 2

        outputs = Conv2DTranspose(filters=1,
                                  kernel_size=self.kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')

        self.decoder.summary()


        #Building the VAE
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae')

        #LOSS
        use_mse=True
        if use_mse:
            reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        else:
            reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                      K.flatten(outputs))

        reconstruction_loss *= self.size_mfcc * (self.min_size_file)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)


        #Compile the VAE
        self.vae.compile(optimizer='rmsprop')
        self.vae.summary()


    def train(self):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        # train the autoencoder

        score = self.vae.fit(self.x_train,
                             epochs=self.epochs,
                             verbose=1,
                             batch_size=self.batch_size,
                             validation_data=(self.x_test, None),
                             callbacks=[es])
        self.vae.save_weights('vae_mlp_mnist.h5')

        score2 = self.vae.evaluate(self.x_test, None, verbose=1)
        print('Score', score.history)
        print('Score', score2)

    def save_array_data(self):
        print("Saving audio array to file")
        np.save(self.file_to_write_X, self.X)
        np.save(self.file_to_write_y, self.y)


    def load_array_data(self):
        print("Loading previous saved data")
        self.file_to_write_X +=".npy"
        self.file_to_write_y += ".npy"
        self.X = np.load(self.file_to_write_X, allow_pickle=True)
        self.y =  np.load(self.file_to_write_y, allow_pickle=True)

    def get_smallest_file(self):


        smallest_file_size=100000000
        smallest_file=""
        for subdir, dirs, files in os.walk(self.path_to_load):
            for file in files:
                filepath = subdir+"/"+ file
                #print(filepath, os.path.getsize(filepath) )
                if os.path.getsize(filepath) <smallest_file_size:
                    #print(filepath, os.path.getsize(filepath) )
                    smallest_file = subdir+"/"+file
                    smallest_file_size = os.path.getsize(filepath)
        print("Chemin du plus petit fichier audio: ", smallest_file)
        return smallest_file



    def clear_dataset(self):

        for root, dirs, files in os.walk(self.path_to_load):
            for name in files:
                path = os.path.join(root, name)
                if os.path.isfile(path):

                    if os.path.getsize(path) < 250000:
                        print("removed: ", name)
                        os.remove(path)  # uncomment this line if you're happy with the set of files to remove

                    if  (name.startswith(".") ):
                        print("removed: ", name)
                        os.remove(path)  # uncomment this line if you're happy with the set of files to remove

                    if not ( name.endswith(".mp3") ): #or  name.endswith(".flac")) :
                        print("removed: ", name)
                        os.remove(path) # uncomment this line if you're happy with the set of files to remove

    def cut_audio(self, save_file_name):

        global counter
        global total_size
        with counter.get_lock():
            counter.value += 1
        print("Done:", counter.value, "/", total_size)

        t2 = 30 * 1000
        if "mp3" in save_file_name:
            newAudio = AudioSegment.from_mp3(save_file_name)
            newAudio = newAudio[:t2]
            save_file_name.replace(".mp3", "")
            newAudio.export(save_file_name, format="mp3")


    def multiprocessed_audio_cut(self, length):
        my_files = []
        for root, dirs, files in os.walk(self.path_to_load):
            for i in files:
                my_files.append(os.path.join(root, i))

        #print(my_files)
        global total_size
        total_size=len(my_files)

        pool = mp.Pool(min(mp.cpu_count(), len(my_files)))  # number of workers
        pool.map(self.cut_audio, my_files, chunksize=1)
        pool.close()


counter = Value('i', 0)
total_size = Value('i', 0)
if __name__ == '__main__':

    vae = CVae()

    #CLEAR DATASET
    #vae.clear_dataset()
    vae.multiprocessed_audio_cut(30)

    '''
    process_audio = True
    if process_audio:
        vae.load_data()
        vae.process_data()
        vae.save_array_data()
    else:
        vae.load_array_data()
        vae.split_data()
        vae.compile()
        vae.train()
        vae.plot_results()
    '''