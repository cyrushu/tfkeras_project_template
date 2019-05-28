"""
This Module is not ready since buggy for tf1.13 and tf2.0
tf1.13 not suitable for keras model save for sess.run
tf2.0 not working as well, waiting for tf2.0 ready for this module
"""

import tensorflow as tf


class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    """
    Callback wraping `tf.train.CheckpointManager`.

    Restores previous checkpoint `on_train_begin`

    Example usage:
    ```python
    model = get_model(...)
    model.compile(optimizer=optimizer, ...)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, '/tmp/my_model', max_to_keep=5)
    callback = CheckpointManagerCallback(checkpoint, manager, period=1)

    model.fit(..., callbacks=[callbacks])
    ```
    """

    def __init__(self, model, config, **kwargs):
        config_callbacks = config["callbacks"]
        self._checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        self._manager = tf.train.CheckpointManager(
            self._checkpoint,
            config_callbacks["checkpoint_dir"],
            max_to_keep=config_callbacks["checkpoint_max_to_keep"])
        self._period = config_callbacks["checkpoint_save_every"]
        self._save_on_train_end = config_callbacks[
            "checkpoint_save_on_train_end"]
        self._restored = False
        self._epoch_count = None
        self._last_save = None

    def _on_begin(self):
        if not self._restored:
            self.restore()

    def restore(self, save_path=None):
        if save_path is None:
            save_path = self._manager.latest_checkpoint
        self._checkpoint.restore(save_path)
        self._restored = True

    def on_train_begin(self, logs=None):
        self._on_begin()

    def on_test_begin(self, logs=None):
        self._on_begin()

    def on_predict_begin(self, logs=None):
        self._on_begin()

    def on_epoch_end(self, epoch, logs=None):
        epochs_finished = epoch + 1
        self._epoch_count = epochs_finished
        if epochs_finished % self._period == 0:
            self._save()

    def on_train_end(self, logs=None):
        if self._save_on_train_end:
            self._save_final()

    def _save(self):
        if self._epoch_count is None:
            return
        if self._last_save != self._epoch_count:
            self._manager.save(self._epoch_count)
            self._last_save = self._epoch_count

    def _save_final(self):
        return self._save()
