import os

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential


class Autoencoder:
    """Autoencoder class for dimensionality reduction of input data.

    The Autoencoder class implements an autoencoder architecture for dimensionality
    reduction of input data. The class includes methods for training, transforming,
    saving, and loading the autoencoder model.
    """

    def __init__(self, X, latent_dim=10, load_model=False):
        """Initialize the Autoencoder class.

        Arguments:
        X -- pandas DataFrame or numpy array, the input data
        latent_dim -- int, the dimension of the latent space (default 10)
        load_model -- bool, whether to load a pre-trained model (default False)
        """
        self.X = X
        self.latent_dim = latent_dim
        self.input_dim = X.shape[1]
        if load_model:
            self.load_model()
        else:
            self.encoder = self._build_encoder()
            self.decoder = self._build_decoder()
            self.autoencoder = self._build_autoencoder()
            self.scaler = self._build_scaler()

    def save_model(self, model_dir="./models"):
        """Save the autoencoder model.

        Arguments:
        model_dir -- str, the directory to save the model (default "./models")
        """
        tf.keras.models.save_model(
            self.encoder, os.path.join(model_dir, "encoder.h5"), save_format="h5"
        )
        tf.keras.models.save_model(
            self.decoder, os.path.join(model_dir, "decoder.h5"), save_format="h5"
        )
        tf.keras.models.save_model(
            self.autoencoder,
            os.path.join(model_dir, "autoencoder.h5"),
            save_format="h5",
        )
        joblib.dump(self.scaler, os.path.join(model_dir, "autoencoder_scaler.pkl"))

    def load_model(self, model_dir="./models"):
        """Load the autoencoder model.

        Arguments:
        model_dir -- str, the directory to load the model from (default "./models")
        """
        self.encoder = tf.keras.models.load_model(
            os.path.join(model_dir, "encoder.h5"), compile=False
        )
        self.decoder = tf.keras.models.load_model(
            os.path.join(model_dir, "decoder.h5"), compile=False
        )
        self.autoencoder = tf.keras.models.load_model(
            os.path.join(model_dir, "autoencoder.h5"), compile=False
        )
        self.scaler = joblib.load(os.path.join(model_dir, "autoencoder_scaler.pkl"))

    def _build_encoder(self):
        """Build the encoder model.

        Returns:
        encoder -- tensorflow.keras.models.Sequential, the encoder model
        """
        encoder = Sequential(
            [
                Dense(128, activation="relu", input_shape=(self.input_dim,)),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(self.latent_dim, activation="relu"),
            ]
        )
        return encoder

    def _build_decoder(self):
        """Build the decoder model.

        Returns:
        decoder -- tensorflow.keras.models.Sequential, the decoder model
        """
        decoder = Sequential(
            [
                Dense(64, activation="relu", input_shape=(self.latent_dim,)),
                Dense(128, activation="relu"),
                Dense(256, activation="relu"),
                Dense(self.input_dim, activation=None),
            ]
        )
        return decoder

    def _build_autoencoder(self):
        """Build the autoencoder model by combining the encoder and decoder models.

        Returns:
        autoencoder -- tensorflow.keras.models.Model, the autoencoder model
        """
        autoencoder = Model(
            inputs=self.encoder.input, outputs=self.decoder(self.encoder.output)
        )
        autoencoder.compile(loss="mse", optimizer="adam")
        return autoencoder

    def _build_scaler(self):
        """Build the scaler for transforming the input data.

        Returns:
        scaler -- sklearn.preprocessing.StandardScaler, the scaler
        """
        X_np = self.X.to_numpy()
        scaler = StandardScaler()
        scaler.fit(X_np)
        return scaler

    def fit(self, epochs=1000):
        """Train the autoencoder model.

        Arguments:
        epochs -- int, the number of training epochs (default 1000)

        Returns:
        history -- tensorflow.python.keras.callbacks.History, the training history
        """
        X_np = self.scaler.transform(self.X.to_numpy())
        self.history = self.autoencoder.fit(
            X_np, X_np, epochs=epochs, batch_size=16, verbose=1
        )

    def transform(self, X_np=None):
        """Transform the input data into the latent space.

        Arguments:
        X_np -- numpy array, the input data to transform (default None, uses self.X)

        Returns:
        latent_representation -- numpy array, the transformed data in the latent space
        """
        if X_np:
            X_np = self.scaler.transform(X_np)
        else:
            X_np = self.scaler.transform(self.X.to_numpy())
        return self.encoder.predict(X_np)
