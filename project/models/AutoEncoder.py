from tensorflow.keras import Input
from tensorflow.keras.layers import Reshape, Dense, Flatten
from tensorflow.keras.models import Model


def autoencoder(inputShape, latent_size):
    inLayer = Input(inputShape)
    encoded = Flatten()(inLayer)

    encoded_shape = encoded.shape[1]
    encoded = Dense(latent_size)(encoded)
    net_t = Dense(encoded_shape)(encoded)
    net_t = Reshape(inLayer.shape[1:])(net_t)

    ae = Model(inLayer, net_t)
    encoder = Model(inLayer, encoded)

    return ae, encoder
