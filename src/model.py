import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization
import numpy as np

class AttentionLayer(Layer):
    """Self-Attention Mechanism for NLP"""
    def __init__(self, attention_dim=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], self.attention_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(self.attention_dim,),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(name='att_u',
                                 shape=(self.attention_dim, 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Attention scoring
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.squeeze(ait, -1)
        ait = tf.nn.softmax(ait)
        
        # Apply attention weights
        ait = tf.expand_dims(ait, -1)
        weighted_input = inputs * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def create_cnn_bilstm_attention(vocab_size, max_len, embedding_dim=300, 
                               embedding_matrix=None, trainable_embeddings=True):
    """
    Create CNN-BiLSTM model with Attention mechanism
    """
    model = tf.keras.Sequential()
    
    # 1. Embedding Layer with FastText
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           weights=[embedding_matrix],
                           input_length=max_len,
                           trainable=trainable_embeddings,
                           mask_zero=True))
    else:
        model.add(Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=max_len,
                           mask_zero=True))
    
    # 2. CNN Layers for local feature extraction
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    
    # 3. Bidirectional LSTM for sequence understanding
    model.add(Bidirectional(LSTM(64, return_sequences=True, 
                                 dropout=0.2, recurrent_dropout=0.1)))
    
    # 4. Attention Layer
    model.add(AttentionLayer(attention_dim=64))
    
    # 5. Dense Layers
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    
    # 6. Output Layer (Regression for -10 to +10)
    model.add(Dense(1, activation='linear'))
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate settings
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0  # Gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error for regression
        metrics=[
            'mae',  # Mean Absolute Error
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
        ]
    )
    # Custom objects için
custom_objects = {
    'AttentionLayer': AttentionLayer
}

# Model oluşturma fonksiyonu (basitleştirilmiş)
def create_simplified_model(vocab_size, max_len, embedding_dim=100):
    """Basitleştirilmiş model (FastText olmadan)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            mask_zero=True
        ),
        
        # CNN katmanları
        tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        # BiLSTM
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)
        ),
        
        # Attention
        AttentionLayer(attention_dim=32),
        
        # Dense katmanları
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(16, activation='relu'),
        
        # Çıktı katmanı
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    return model