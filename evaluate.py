#import libraries
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
#extract features

def read_file(filename):
  file = open(filename, 'r')
  text = file.read()
  file.close()
  return text

def load_set(filename):
    text = read_file(filename)
    dataset = list()

    for line in text.split("\n"):
        if (len(line) < 2):
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

        if (id in dataset):
            if id not in descriptions:
                descriptions[id] = list()
            desc = 'startseq ' + ' '.join(desc) + ' endseq'

            descriptions[id].append(desc)
    return descriptions


def load_photo_features(filename, dataset):
    encoded_features = pickle.load(open(filename, 'rb'))

    features = {k: encoded_features[k] for k in dataset}
    return features


def extract_features(filename):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def idx_to_word(integer, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == integer:
            return word
    return None
#create caption for an image
def create_captions(model, tokenizer, photo, max_length):
    in_text = 'startseq'

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        output = model.predict([photo, sequence], verbose=0)
        output = np.argmax(output)
        predicted_word = idx_to_word(output, tokenizer)
        if predicted_word is None:
            break
        in_text += ' ' + predicted_word
        if predicted_word == 'endseq':
            break
    return in_text



def evaluate_model(model, descriptions, photos, tokenizer, max_length):
  actual, predicted = list(), list()
  # step over the whole set
  for key, desc_list in descriptions.items():
    # generate description
    yhat = create_captions(model, tokenizer, photos[key], max_length)
    # store actual and predicted
    references = [d.split() for d in desc_list]
    actual.append(references)
    predicted.append(yhat.split())
  # calculate BLEU score
  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


tokenizer = pickle.load(open('./FYP_dataset/tokenizer.pkl', 'rb'))


filename = './FYP_dataset/Flicker8k_text/Flickr_8k.devImages.txt'
test_text = load_set(filename)
test_descriptions = load_clean_descriptions('./FYP_dataset/descriptions.txt', test_text)
test_features = load_photo_features("./FYP_dataset/features.pkl", test_text)
#X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

model = load_model('./models/model_9.h5')


max_length = 34

#evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)


photo = extract_features("./FYP_dataset/Flicker8k_Dataset/143552697_af27e9acf5.jpg")

description = create_captions(model, tokenizer, photo, max_length)
print(description)
