import os, sys
import math
import numpy as np
import random

import matplotlib.pyplot as plt
import sentencepiece as spm
import tensorflow as tf

import preprocess as pr
import bert


# random seed
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# default path
abs_path = os.path.abspath('')
abs_path = os.path.dirname(abs_path)
corpus_file = abs_path + "/data/kowiki.txt"
pretrain_json_path = abs_path + '/data/bert_pre_train.json'
vocab_size = 8000
prefix = abs_path + f"/data/ko_{vocab_size}"
print(prefix)
model_name = "bert_scriptrun_8k.hdf5"



def get_vocab(corpus_file, prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        f"--input={corpus_file} --model_prefix={prefix} --vocab_size={vocab_size + 7} --model_type=bpe --max_sentence_length=999999 --pad_id=0 --pad_piece=[PAD] --unk_id=1 --unk_piece=[UNK] --bos_id=2 --bos_piece=[BOS] --eos_id=3 --eos_piece=[EOS] --user_defined_symbols=[SEP],[CLS],[MASK]")

    # vocab loading
    vocab = spm.SentencePieceProcessor()
    vocab.load(f"{prefix}.model")
    
    return vocab


# preprocessing
def get_dataset(abs_path, vocab, corpus_file, pretrain_json_path):
    
    
    count = 128000

#     pr.make_pretrain_data(vocab, corpus_file, pretrain_json_path, 128)
    inputs, labels = pr.load_pre_train_data(
        vocab, pretrain_json_path, 128, count=count)
    
    return inputs, labels


# processing
def process(vocab, inputs, labels, model_name):
    config = bert.Config({"d_model": 256, "n_head": 4, "d_head": 64, "dropout": 0.1, "d_ff": 1024, "layernorm_epsilon": 0.001, "n_layer": 3, "n_seq": 256, "n_vocab": 0, "i_pad": 0})
    config.n_vocab = len(vocab)
    config.i_pad = vocab.pad_id()

    # compute lr 
    test_schedule = bert.CosineSchedule(train_steps=4000, warmup_steps=500)
    lrs = []
    for step_num in range(4000):
        lrs.append(test_schedule(float(step_num)).numpy())

    pre_train_model = bert.build_model_pre_train(config)
    pre_train_model.summary()

    input_length = len(inputs[0])

    epochs = 10
    batch_size = 64

    # optimizer
    train_steps = math.ceil(input_length / batch_size) * epochs
    print("train_steps:", train_steps)
    learning_rate = bert.CosineSchedule(train_steps=train_steps, warmup_steps=max(100, train_steps // 10))
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # compile
    pre_train_model.compile(
        loss=(tf.keras.losses.sparse_categorical_crossentropy, bert.lm_loss),
        optimizer=optimizer,
        metrics={"nsp": "acc", "mlm": bert.lm_acc})

    # Q. 모델을 학습시키고, 내용을 history에 담아주세요.
    # save weights callback
    save_weights = tf.keras.callbacks.ModelCheckpoint(
        f"{model_dir}/{model_name}", monitor="mlm_lm_acc", 
        verbose=1, save_best_only=True, mode="max", 
        save_freq="epoch",
    #     save_freq=10,
        save_weights_only=True)

    # train
    # 모델 인자에는 inputs, labels, epochs, batch size, callback 이 필요해요.
    history = pre_train_model.fit(
        inputs, labels,
        steps_per_epoch=50,
    #     epochs=epochs,
        epochs=5,
        verbose=2,
        batch_size=batch_size,
        callbacks=[save_weights])
    
    return history


def plot_history(history, model_name):

    # training result
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['nsp_loss'], 'b-', label='nsp_loss')
    plt.plot(history.history['mlm_loss'], 'r--', label='mlm_loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['nsp_acc'], 'g-', label='nsp_acc')
    plt.plot(history.history['mlm_lm_acc'], 'k--', label='mlm_acc')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig(f"./{model_name}_result.png")
    plt.show()


vocab = get_vocab(corpus_file, prefix, vocab_size)
inputs, labels = get_dataset(abs_path, vocab, corpus_file, pretrain_json_path)
history = process(vocab, inputs, labels, model_name)
plot_history(history, model_name)
