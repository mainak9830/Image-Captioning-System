# -*- coding: utf-8 -*-
"""FYP_2022.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kSkWYvXFNSXaJ52auW6JYY2e3OitxBDd
"""

#import libraries
from matplotlib import pyplot as plt
from model import create_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from nltk.translate.bleu_score import corpus_bleu
import os
import pickle
import string
from keras.models import load_model
#feature extraction using CNN
def feature_extract(directory):
  # model
  model = VGG16()
  model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

  features = dict()

  for name in os.listdir(directory):
    file = directory + '/' + name
    image = load_img(file, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = name.split('.')[0]

    features[image_id] = feature
  return features

directory = "/FYP_dataset/Flicker8k_Dataset"
#features = feature_extract(directory)
#pickle.dump(features, open('/FYP_dataset/features.pkl', 'wb'))

#reading the file
def read_file(filename):
  file = open(filename, 'r')
  text = file.read()
  file.close()
  return text

filename = "./FYP_dataset/Flicker8k_text/Flickr8k.token.txt"
text = read_file(filename)

#create mappings from image to text
def create_mappings(text):
  mapping = dict()

  #separate each rows
  for line in text.split("\n"):
    #split by tab
    tokens = line.split("\t")
    if(len(tokens) < 2):
      continue
    id, details = tokens[0], tokens[1]
    id = id.split('.')[0]
    
    if id not in mapping:
      mapping[id] = list()
    mapping[id].append(details)
  return mapping

mappings = create_mappings(text)

len(mappings)

#clean the text
def clean_text(map):
  table = str.maketrans('', '', string.punctuation)

  for id, text_list in map.items():
    for i in range(len(text_list)):
      desc = text_list[i]
      desc = desc.split()
      desc = [word.lower() for word in desc]
      desc = [word.translate(table) for word in desc]
      desc = [word for word in desc if len(word)>1]
      desc = [word for word in desc if word.isalpha()]
      text_list[i] =  ' '.join(desc)

clean_text(mappings)

#vocabulary creation
def to_vocabulary(mapping):
  dictionary = set()
  for key in mapping.keys():
    [dictionary.update(d.split()) for d in mapping[key]]
  return dictionary

vocabulary = to_vocabulary(mappings)

len(vocabulary)

def save_cleaned_text(mappings, output):
  lines = list()
  for id, text_list in mappings.items():
    for text in text_list:
      lines.append(id + ' ' + text)
  
  new_text = '\n'.join(lines)
  file = open(output, 'w')
  file.write(new_text)
  file.close()

save_cleaned_text(mappings, './FYP_dataset/descriptions.txt')

def load_set(filename):
  text = read_file(filename)
  dataset = list()

  for line in text.split("\n"):
    if(len(line) < 2):
      continue
    id = line.split('.')[0]
    dataset.append(id)
  return set(dataset)

def load_clean_descriptions(filename, dataset):
  text = read_file(filename)
  descriptions = dict()

  for line in text.split("\n"):

    tokens = line.split()

    id, desc = tokens[0], tokens[1:]

    if(id in dataset):
      if id not in descriptions:
        descriptions[id] = list()
      desc = 'startseq ' + ' '.join(desc) + ' endseq'

      descriptions[id].append(desc)
  return descriptions
      


def load_photo_features(filename, dataset):
  encoded_features = pickle.load(open(filename, 'rb'))

  features = {k: encoded_features[k] for k in dataset}
  return features



from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

def create_dataset(descriptions):
  rows = list()
  for key in descriptions.keys():
    [rows.append(d) for d in descriptions[key]]
  return rows

def create_tokenizer(descriptions):
  rows = create_dataset(descriptions)
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(rows)
  return tokenizer

def max_length(descriptions):
  rows = create_dataset(descriptions)
  return max(len(d.split()) for d in rows)

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
  X1, X2, y = list(), list(), list()
  # walk through each description for the image
  for desc in desc_list:
      # encode the sequence
      seq = tokenizer.texts_to_sequences([desc])[0]
      # split one sequence into multiple X,y pairs
      for i in range(1, len(seq)):
          # split into input and output pair
          in_seq, out_seq = seq[:i], seq[i]
          # pad input sequence
          in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
          # encode output sequence
          out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
          # store
          X1.append(photo)
          X2.append(in_seq)
          y.append(out_seq)
  return array(X1), array(X2), array(y)

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
            yield [in_img, in_seq], out_word

import numpy as np




#train dataset
filename = "./FYP_dataset/Flicker8k_text/Flickr_8k.trainImages.txt"

train_text = load_set(filename)

train_descriptions = load_clean_descriptions("./FYP_dataset/descriptions.txt", train_text)
train_features = load_photo_features("./FYP_dataset/features.pkl", train_text)
print('Descriptions: train=%d' % len(train_descriptions))
print('Photos: train=%d' % len(train_features))
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max_length(train_descriptions)
print('Vocabulary Size: %d' % vocab_size)
pickle.dump(tokenizer, open('./FYP_dataset/tokenizer.pkl', 'wb'))

model = create_model(vocab_size, max_length)
epochs = 10
steps = len(train_descriptions)
import time




start_time = time.time()
loss = []
for i in range(epochs):
  # create the data generator
  generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
  # fit for one epoch
  history = model.fit_generator(generator, epochs=1, steps_per_epoch=1500, verbose=1)
  loss.append(history.history['loss'])
  # save model
  model.save('./models/model_' + str(i) + '.h5')

end_time = time.time()

print((end_time-start_time)/60)
plt.plot(loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
plt.savefig('loss.png', bbox_inches='tight')


#test dataset

filename = './FYP_dataset/Flicker8k_text/Flickr_8k.devImages.txt'
test_text = load_set(filename)
test_descriptions = load_clean_descriptions('./FYP_dataset/descriptions.txt', test_text)
test_features = load_photo_features("./FYP_dataset/features.pkl", test_text)
#X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

model = load_model('./models/model_9.h5')
