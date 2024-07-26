from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_dataset(vocab_size, max_len):
    (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=vocab_size, test_split=0.2)

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)


def get_lstm(vocab_size):
    embedding_dim = 128
    hidden_units = 128
    num_classes = 46

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(hidden_units))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model

def fit(vocab_size=1000, max_len=100, epochs=30):
    dataset = get_dataset(vocab_size, max_len)
    (X_train, y_train), (X_test, y_test) = dataset
    model = get_lstm(vocab_size)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    history = model.fit(
        X_train, y_train, 
        batch_size=128, epochs=epochs, 
        callbacks=[es, mc], 
        validation_data=(X_test, y_test))
    
    return dataset, history

def load_saved_model(file_path):
    return load_model(file_path)

def evaluate_model(model, X_test, y_test):
    acc = model.evaluate(X_test, y_test)[1]
    print("\n 테스트 정확도: %.4f" % (acc))
    return acc

def plot_result(history, label):
    epochs = range(1, len(history.history['acc']) + 1)
    plt.plot(epochs, history.history[label])
    plt.plot(epochs, history.history[f'val_{label}'])
    plt.title(f'model {label}')
    plt.ylabel(label)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f"lstm_{label}.png")
