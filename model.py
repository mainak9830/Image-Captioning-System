#import libraries
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


def create_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    lr1 = Dropout(0.5)(inputs1)
    lr2 = Dense(256, activation="relu")(lr1)

    inputs2 = Input(shape=(max_length,))
    dcr1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    dcr2 = Dropout(0.5)(dcr1)
    dcr3 = LSTM(256)(dcr2)

    #linear model
    f1 = add([lr2, dcr3])
    f2 = Dense(256, activation="relu")(f1)
    out = Dense(vocab_size, activation="softmax")(f2)

    model = Model(inputs=[inputs1, inputs2], outputs=out)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    model.summary()
    #plot_model(model, to_file="model.png", show_shapes=True)
    return model

create_model(256, 34)

