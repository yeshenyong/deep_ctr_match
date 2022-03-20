
from tensorflow.python.keras.layers import LSTM, Lambda, Layer

class WeightedSequenceLayer(Layer):
    """The WeightedSequenceLayer is used to apply weight score on variable-length sequence feature/multi-value feature.

    Input shape
        - A list of two tensor [seq_value, seq_len, seq_weight]

    """




