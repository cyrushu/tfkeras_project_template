import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import os
from utils.args import Parser
from utils.config import parse_user_config
from utils import utils
from utils import factory
import yaml


def main():
    args = Parser()
    config = parse_user_config(args.config)
    # adapt this if using `channels_first` image data format
    train_dg, test_dg = factory.create("data_generator." +
                                       config["data_generator"]["name"])(
                                           **config["data_generator"])
    if config["model"]["parameters"] is None:
        model, _ = factory.create("models." + config["model"]["name"])()
    else:
        model, _ = factory.create("models." + config["model"]["name"])(
            **config["model"]["parameters"])

    optimizer = factory.create(
        config["optimizer"]["name"])(**config["optimizer"]["parameters"])
    model.compile(loss="mse", optimizer=optimizer)

    saver_callback = factory.create(
        "tensorflow.keras.callbacks.ModelCheckpoint")(
            os.path.join(
                utils.make_valid_dir(config["callbacks"]["checkpoint_dir"]),
                config["pipeline"]["name"] + "{epoch}.ckpt"),
            verbose=1,
            period=config["callbacks"]["checkpoint_save_every"])

    model_yaml = model.to_yaml()
    folder_save_dir = config["callbacks"]["folder_name"]
    with open(os.path.join(folder_save_dir, "model.yaml"), "w") as f:
        f.write(model_yaml)
    with open(os.path.join(folder_save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    #  saver_callback = factory.create(
    #  "callbacks.CheckpointManagerCallback.CheckpointManagerCallback")(
    #  model, config)
    #  saver_callback.restore()

    model.fit_generator(
        train_dg,
        epochs=config["train"]["epochs"],
        callbacks=[
            saver_callback,
            TensorBoard(log_dir=config["callbacks"]["tensorboard_log_dir"])
        ])


if __name__ == "__main__":
    main()
