from bert_runner import BERTRunner

r = BERTRunner()
r.make_vocab()
r.make_dataset(128000)

r.set_config(
    d_model=128,
    n_head=4,
    d_head=32,
    dropout=0.1,
    d_ff=512,
    layernorm_epsilon=0.001,
    n_layer=2,
    n_seq=128,
    n_vocab=0,
    i_pad=0
)

r.build(epochs=10)
r.train()
r.plot_history()
