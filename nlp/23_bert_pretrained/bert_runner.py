import os, sys
import math
import numpy as np
import random

import matplotlib.pyplot as plt
import sentencepiece as spm
import tensorflow as tf

import preprocess as pr
import bert


class BERTRunner:
    
    def __init__(self, vocab_size=8000):
        # random seed
        random_seed = 1234
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        # default path
        self.abs_path = os.path.abspath('')
        self.abs_path = os.path.dirname(self.abs_path)
        self.corpus_file = self.abs_path + "/data/kowiki.txt"
        self.pretrain_json_path = self.abs_path + '/data/bert_pre_train.json'
        self.vocab_size = vocab_size
        self.prefix = self.abs_path + f"/data/ko_{self.vocab_size}"
        print(self.prefix)
        self.model_name = f"bert_{self.vocab_size}.hdf5"
        self.set_config()
        

    def make_vocab(self):
        spm.SentencePieceTrainer.train(
            f"--input={self.corpus_file} --model_prefix={self.prefix} --vocab_size={self.vocab_size + 7} --model_type=bpe --max_sentence_length=999999 --pad_id=0 --pad_piece=[PAD] --unk_id=1 --unk_piece=[UNK] --bos_id=2 --bos_piece=[BOS] --eos_id=3 --eos_piece=[EOS] --user_defined_symbols=[SEP],[CLS],[MASK]")

        # vocab loading
        self.vocab = spm.SentencePieceProcessor()
        self.vocab.load(f"{self.prefix}.model")


    # preprocessing
    def make_dataset(self, count=128000):
        
        pr.make_pretrain_data(self.vocab, self.corpus_file, self.pretrain_json_path, 128)
        self.inputs, self.labels = pr.load_pre_train_data(
            self.vocab, self.pretrain_json_path, 128, count=count)
        
        
    def set_config(self,
                   d_model=256,
                   n_head=4,
                   d_head=64,
                   dropout=0.1,
                   d_ff=1024,
                   layernorm_epsilon=0.001,
                   n_layer=3,
                   n_seq=256,
                   n_vocab=0,
                   i_pad=0
                  ):
        
        self.config = bert.Config({
            "d_model": d_model,
            "n_head": n_head,
            "d_head": d_head,
            "dropout": dropout,
            "d_ff": d_ff,
            "layernorm_epsilon": layernorm_epsilon,
            "n_layer": n_layer,
            "n_seq": n_seq,
            "n_vocab": n_vocab,
            "i_pad": i_pad
        })



    # processing
    def build(self, epochs=10, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.config.n_vocab = len(self.vocab)
        self.config.i_pad = self.vocab.pad_id()

        # compute lr 
        test_schedule = bert.CosineSchedule(train_steps=4000, warmup_steps=500)
        lrs = []
        for step_num in range(4000):
            lrs.append(test_schedule(float(step_num)).numpy())

        self.pre_train_model = bert.build_model_pre_train(self.config)
        self.pre_train_model.summary()

        input_length = len(self.inputs[0])

        # optimizer
        train_steps = math.ceil(input_length / batch_size) * epochs
        print("train_steps:", train_steps)
        learning_rate = bert.CosineSchedule(train_steps=train_steps, warmup_steps=max(100, train_steps // 10))
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # compile
        self.pre_train_model.compile(
            loss=(tf.keras.losses.sparse_categorical_crossentropy, bert.lm_loss),
            optimizer=optimizer,
            metrics={"nsp": "acc", "mlm": bert.lm_acc})

        # Q. 모델을 학습시키고, 내용을 history에 담아주세요.
        # save weights callback
        self.save_weights = tf.keras.callbacks.ModelCheckpoint(
            f"{self.abs_path}/data/{self.model_name}", monitor="mlm_lm_acc", 
            verbose=1, save_best_only=True, mode="max", 
            save_freq="epoch",
        #     save_freq=10,
            save_weights_only=True)

        
    def train(self):
        # train
        # 모델 인자에는 inputs, labels, epochs, batch size, callback 이 필요해요.
        self.history = self.pre_train_model.fit(
            self.inputs, self.labels,
#             steps_per_epoch=50,
            epochs=self.epochs,
#             epochs=5,
            verbose=1,
            batch_size=self.batch_size,
            callbacks=[self.save_weights])


    def plot_history(self):
        
        history = self.history
        model_name = self.model_name.split(".")[0]

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


    def process(self):
        self.make_vocab()
        self.make_dataset(self.abs_path, self.vocab, self.corpus_file, self.pretrain_json_path)
        self.build()
        self.plot_history()

        