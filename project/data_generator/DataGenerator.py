import numpy as np
from tensorflow import keras
from sklearn import model_selection
import h5py


def get_train_val_generator(*args, **kwargs):
    h5f = h5py.File(kwargs.get("datah5"))
    data_len = len(h5f["channel"])
    indexes_train, indexes_test = model_selection.train_test_split(
        np.arange(data_len), test_size=kwargs.get("train_test_traio",0.1))
    train_dg = DataGenerator(h5f, indexes_train, **kwargs)
    test_dg = DataGenerator(h5f, indexes_test, **kwargs)
    return train_dg, test_dg


class DataGenerator(keras.utils.Sequence):
    """DataGenerator"""

    def __init__(self, h5f, indexes, **kwargs):
        self.h5f = h5f
        self.indexes_origin = indexes
        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.indexes_origin.shape[0]) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        indexes.sort()
        channel = self.h5f["channel"][indexes.tolist()]
        channel_gaussian = self.h5f["channel_gaussian"][indexes.tolist()]

        channel = np.moveaxis(channel, 1, -1)
        channel_gaussian = np.moveaxis(channel_gaussian, 1, -1)
        return (channel, channel_gaussian), None

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.indexes_origin
        if self.shuffle:
            np.random.shuffle(self.indexes)
