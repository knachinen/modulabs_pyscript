

## Process - run_bert.py

- Preprocessing
	- get_vocab() - SentencePiece
	- get_dataset() - Mask, NSP pair
- Process
	- process() - building a model, training
- Result
	- plot_history() - nsp/mlm loss, nsp/mlm accuracy


## Modules

### Preprocessing - preprocess.py

### BERT model - bert.py



## Results

![](bert_pretrained_8k_result.png)

```
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
enc_tokens (InputLayer)         [(None, None)]       0                                            
__________________________________________________________________________________________________
segments (InputLayer)           [(None, None)]       0                                            
__________________________________________________________________________________________________
bert (BERT)                     ((None, 256), (None, 4485632     enc_tokens[0][0]                 
                                                                 segments[0][0]                   
__________________________________________________________________________________________________
pooled_nsp (PooledOutput)       (None, 2)            66304       bert[0][0]                       
__________________________________________________________________________________________________
nsp (Softmax)                   (None, 2)            0           pooled_nsp[0][0]                 
__________________________________________________________________________________________________
mlm (Softmax)                   (None, None, 8007)   0           bert[0][1]                       
==================================================================================================
Total params: 4,551,936
Trainable params: 4,551,936
Non-trainable params: 0
__________________________________________________________________________________________________
train_steps: 20000
Epoch 1/10
2000/2000 [==============================] - 250s 123ms/step - loss: 15.5390 - nsp_loss: 0.6526 - mlm_loss: 14.8864 - nsp_acc: 0.5881 - mlm_lm_acc: 0.3297

Epoch 00001: mlm_lm_acc improved from -inf to 0.32970, saving model to bert_prj_8k.hdf5
Epoch 2/10
2000/2000 [==============================] - 246s 123ms/step - loss: 13.5583 - nsp_loss: 0.6146 - mlm_loss: 12.9437 - nsp_acc: 0.6281 - mlm_lm_acc: 0.3662

Epoch 00002: mlm_lm_acc improved from 0.32970 to 0.36615, saving model to bert_prj_8k.hdf5
Epoch 3/10
2000/2000 [==============================] - 247s 124ms/step - loss: 12.6591 - nsp_loss: 0.6079 - mlm_loss: 12.0512 - nsp_acc: 0.6380 - mlm_lm_acc: 0.3736

Epoch 00003: mlm_lm_acc improved from 0.36615 to 0.37355, saving model to bert_prj_8k.hdf5
Epoch 4/10
2000/2000 [==============================] - 247s 123ms/step - loss: 11.5271 - nsp_loss: 0.6034 - mlm_loss: 10.9237 - nsp_acc: 0.6465 - mlm_lm_acc: 0.3907

Epoch 00004: mlm_lm_acc improved from 0.37355 to 0.39067, saving model to bert_prj_8k.hdf5
Epoch 5/10
2000/2000 [==============================] - 247s 123ms/step - loss: 10.9517 - nsp_loss: 0.5983 - mlm_loss: 10.3533 - nsp_acc: 0.6568 - mlm_lm_acc: 0.4014

Epoch 00005: mlm_lm_acc improved from 0.39067 to 0.40142, saving model to bert_prj_8k.hdf5
Epoch 6/10
2000/2000 [==============================] - 247s 124ms/step - loss: 10.6201 - nsp_loss: 0.5920 - mlm_loss: 10.0281 - nsp_acc: 0.6700 - mlm_lm_acc: 0.4078

Epoch 00006: mlm_lm_acc improved from 0.40142 to 0.40776, saving model to bert_prj_8k.hdf5
Epoch 7/10
2000/2000 [==============================] - 247s 123ms/step - loss: 10.4012 - nsp_loss: 0.5857 - mlm_loss: 9.8155 - nsp_acc: 0.6810 - mlm_lm_acc: 0.4116

Epoch 00007: mlm_lm_acc improved from 0.40776 to 0.41161, saving model to bert_prj_8k.hdf5
Epoch 8/10
2000/2000 [==============================] - 247s 123ms/step - loss: 10.2550 - nsp_loss: 0.5782 - mlm_loss: 9.6768 - nsp_acc: 0.6952 - mlm_lm_acc: 0.4144

Epoch 00008: mlm_lm_acc improved from 0.41161 to 0.41439, saving model to bert_prj_8k.hdf5
Epoch 9/10
2000/2000 [==============================] - 247s 123ms/step - loss: 10.1644 - nsp_loss: 0.5718 - mlm_loss: 9.5926 - nsp_acc: 0.7056 - mlm_lm_acc: 0.4159

Epoch 00009: mlm_lm_acc improved from 0.41439 to 0.41591, saving model to bert_prj_8k.hdf5
Epoch 10/10
2000/2000 [==============================] - 247s 123ms/step - loss: 10.1212 - nsp_loss: 0.5686 - mlm_loss: 9.5526 - nsp_acc: 0.7097 - mlm_lm_acc: 0.4166

Epoch 00010: mlm_lm_acc improved from 0.41591 to 0.41659, saving model to bert_prj_8k.hdf5
```


```markdown
d_model=128,  
n_head=4,  
d_head=16,  
dropout=0.1,  
d_ff=256,  
layernorm_epsilon=0.001,  
n_layer=2,  
n_seq=128,  
n_vocab=0,  
i_pad=0
```

```
Model: "model_5"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
enc_tokens (InputLayer)         [(None, None)]       0                                            
__________________________________________________________________________________________________
segments (InputLayer)           [(None, None)]       0                                            
__________________________________________________________________________________________________
bert (BERT)                     ((None, 128), (None, 1240832     enc_tokens[0][0]                 
                                                                 segments[0][0]                   
__________________________________________________________________________________________________
pooled_nsp (PooledOutput)       (None, 2)            16768       bert[0][0]                       
__________________________________________________________________________________________________
nsp (Softmax)                   (None, 2)            0           pooled_nsp[0][0]                 
__________________________________________________________________________________________________
mlm (Softmax)                   (None, None, 8007)   0           bert[0][1]                       
==================================================================================================
Total params: 1,257,600
Trainable params: 1,257,600
Non-trainable params: 0
__________________________________________________________________________________________________
train_steps: 20000

In [27]:

r.train()
```

```
Epoch 1/10
2000/2000 [==============================] - 121s 60ms/step - loss: 16.7610 - nsp_loss: 0.6703 - mlm_loss: 16.0907 - nsp_acc: 0.5583 - mlm_lm_acc: 0.2907

Epoch 00001: mlm_lm_acc improved from -inf to 0.29066, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 2/10
2000/2000 [==============================] - 119s 60ms/step - loss: 14.2999 - nsp_loss: 0.6374 - mlm_loss: 13.6625 - nsp_acc: 0.6081 - mlm_lm_acc: 0.3520

Epoch 00002: mlm_lm_acc improved from 0.29066 to 0.35199, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 3/10
2000/2000 [==============================] - 119s 60ms/step - loss: 13.7734 - nsp_loss: 0.6288 - mlm_loss: 13.1446 - nsp_acc: 0.6201 - mlm_lm_acc: 0.3629

Epoch 00003: mlm_lm_acc improved from 0.35199 to 0.36290, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 4/10
2000/2000 [==============================] - 119s 60ms/step - loss: 13.5245 - nsp_loss: 0.6203 - mlm_loss: 12.9042 - nsp_acc: 0.6345 - mlm_lm_acc: 0.3662

Epoch 00004: mlm_lm_acc improved from 0.36290 to 0.36624, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 5/10
2000/2000 [==============================] - 119s 60ms/step - loss: 13.3515 - nsp_loss: 0.6077 - mlm_loss: 12.7438 - nsp_acc: 0.6514 - mlm_lm_acc: 0.3676

Epoch 00005: mlm_lm_acc improved from 0.36624 to 0.36757, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 6/10
2000/2000 [==============================] - 119s 60ms/step - loss: 13.2417 - nsp_loss: 0.6029 - mlm_loss: 12.6389 - nsp_acc: 0.6600 - mlm_lm_acc: 0.3683

Epoch 00006: mlm_lm_acc improved from 0.36757 to 0.36827, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 7/10
2000/2000 [==============================] - 119s 60ms/step - loss: 13.1601 - nsp_loss: 0.5985 - mlm_loss: 12.5616 - nsp_acc: 0.6679 - mlm_lm_acc: 0.3687

Epoch 00007: mlm_lm_acc improved from 0.36827 to 0.36874, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 8/10
2000/2000 [==============================] - 119s 60ms/step - loss: 13.1006 - nsp_loss: 0.5947 - mlm_loss: 12.5059 - nsp_acc: 0.6745 - mlm_lm_acc: 0.3690

Epoch 00008: mlm_lm_acc improved from 0.36874 to 0.36901, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 9/10
2000/2000 [==============================] - 119s 60ms/step - loss: 13.0652 - nsp_loss: 0.5917 - mlm_loss: 12.4735 - nsp_acc: 0.6803 - mlm_lm_acc: 0.3691

Epoch 00009: mlm_lm_acc improved from 0.36901 to 0.36907, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 10/10
2000/2000 [==============================] - 119s 60ms/step - loss: 13.0487 - nsp_loss: 0.5904 - mlm_loss: 12.4583 - nsp_acc: 0.6826 - mlm_lm_acc: 0.3692

Epoch 00010: mlm_lm_acc improved from 0.36907 to 0.36916, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
```



```
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
```

```
Model: "model_6"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
enc_tokens (InputLayer)         [(None, None)]       0                                            
__________________________________________________________________________________________________
segments (InputLayer)           [(None, None)]       0                                            
__________________________________________________________________________________________________
bert (BERT)                     ((None, 128), (None, 1438336     enc_tokens[0][0]                 
                                                                 segments[0][0]                   
__________________________________________________________________________________________________
pooled_nsp (PooledOutput)       (None, 2)            16768       bert[0][0]                       
__________________________________________________________________________________________________
nsp (Softmax)                   (None, 2)            0           pooled_nsp[0][0]                 
__________________________________________________________________________________________________
mlm (Softmax)                   (None, None, 8007)   0           bert[0][1]                       
==================================================================================================
Total params: 1,455,104
Trainable params: 1,455,104
Non-trainable params: 0
__________________________________________________________________________________________________
train_steps: 20000
```

```
Epoch 1/10
2000/2000 [==============================] - 134s 66ms/step - loss: 16.7048 - nsp_loss: 0.6716 - mlm_loss: 16.0333 - nsp_acc: 0.5542 - mlm_lm_acc: 0.2921

Epoch 00001: mlm_lm_acc improved from -inf to 0.29208, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 2/10
2000/2000 [==============================] - 131s 66ms/step - loss: 14.2410 - nsp_loss: 0.6377 - mlm_loss: 13.6033 - nsp_acc: 0.6128 - mlm_lm_acc: 0.3521

Epoch 00002: mlm_lm_acc improved from 0.29208 to 0.35208, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 3/10
2000/2000 [==============================] - 132s 66ms/step - loss: 13.6607 - nsp_loss: 0.6180 - mlm_loss: 13.0427 - nsp_acc: 0.6368 - mlm_lm_acc: 0.3628

Epoch 00003: mlm_lm_acc improved from 0.35208 to 0.36277, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 4/10
2000/2000 [==============================] - 132s 66ms/step - loss: 13.3916 - nsp_loss: 0.6069 - mlm_loss: 12.7846 - nsp_acc: 0.6529 - mlm_lm_acc: 0.3659

Epoch 00004: mlm_lm_acc improved from 0.36277 to 0.36592, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 5/10
2000/2000 [==============================] - 132s 66ms/step - loss: 13.2215 - nsp_loss: 0.5991 - mlm_loss: 12.6224 - nsp_acc: 0.6645 - mlm_lm_acc: 0.3671

Epoch 00005: mlm_lm_acc improved from 0.36592 to 0.36707, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 6/10
2000/2000 [==============================] - 131s 66ms/step - loss: 13.0913 - nsp_loss: 0.5938 - mlm_loss: 12.4975 - nsp_acc: 0.6747 - mlm_lm_acc: 0.3678

Epoch 00006: mlm_lm_acc improved from 0.36707 to 0.36779, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 7/10
2000/2000 [==============================] - 131s 66ms/step - loss: 12.9870 - nsp_loss: 0.5888 - mlm_loss: 12.3982 - nsp_acc: 0.6829 - mlm_lm_acc: 0.3682

Epoch 00007: mlm_lm_acc improved from 0.36779 to 0.36816, saving model to /aiffel/aiffel/data/bert_scriptrun_8k.hdf5
Epoch 8/10
 590/2000 [=======>......................] - ETA: 1:32 - loss: 12.9099 - nsp_loss: 0.5843 - mlm_loss: 12.3255 - nsp_acc: 0.6922 - mlm_lm_acc: 0.3684

In [ ]:

​
```


