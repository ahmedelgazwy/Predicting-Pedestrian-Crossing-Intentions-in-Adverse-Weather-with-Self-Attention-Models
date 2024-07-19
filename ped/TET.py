import action_predict
import tensorflow as tf
import numpy as np




input_array = np.random.rand(10, 10)
sample_encoder = action_predict.Encoder(num_layers=4,d_model=10,num_heads=8,dff=2048,vocab_size=8500)
sample_encoder_output = sample_encoder(input_array, training=False)
print(input_array.shape)
print(sample_encoder_output.shape) 