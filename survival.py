import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sksurv.datasets import load_breast_cancer
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

from autoencoder import Autoencoder


class SurvivalAnalysis:
    """SurvivalAnalysis class for training a survival analysis model and predicting survival functions.

    Attributes:
    X -- pandas dataframe, the input features
    y -- pandas dataframe, the target values (survival times and event indicators)
    latent_dim -- int, the dimension of the latent space (default 10)
    X_train -- pandas dataframe, the training data for the input features
    X_test -- pandas dataframe, the test data for the input features
    y_train -- pandas dataframe, the training data for the target values
    y_test -- pandas dataframe, the test data for the target values
    autoencoder -- autoencoder.Autoencoder, the autoencoder used for dimensionality reduction
    model -- sksurv.ensemble.GradientBoostingSurvivalAnalysis, the survival analysis model
    explainer -- shap.Explainer, the SHAP explainer for the survival analysis model

    Methods:
    save_model -- save the survival analysis model and autoencoder to disk
    run -- run the survival analysis, including training and evaluating the model
    predict_survival_function -- predict the survival function for a new sample
    plot_shap_waterfall -- plot the SHAP waterfall plot for a new sample
    """

    def __init__(self, X, y, latent_dim=10, load_model=False, model_dir="./models"):
        """Initialize the SurvivalAnalysis object.

        Arguments:
        X -- pandas.DataFrame, the feature matrix
        y -- pandas.DataFrame, the target matrix (survival time, censoring status)

        Keyword Arguments:
        latent_dim -- int, the number of dimensions to use in the autoencoder (default 10)
        load_model -- bool, whether to load a pre-trained model from disk (default False)
        model_dir -- str, the directory to store the models in (default "./models")

        Returns: None
        """
        self.X = X
        self.y = y
        self.latent_dim = latent_dim
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )

        # Load from disk in case load_model is set, else create a new model.
        self.autoencoder = Autoencoder(self.X, self.latent_dim, load_model)
        if load_model:
            self.model = joblib.load(os.path.join(model_dir, "survival.pkl"))
            self.explainer = joblib.load(os.path.join(model_dir, "explainer.pkl"))
        else:
            self.model = GradientBoostingSurvivalAnalysis(
                n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
            )
            self.explainer = shap.Explainer(
                self.model.predict,
                self.X_train,
                feature_names=list(self.X_train.columns),
            )

    def save_model(self, model_dir="./models"):
        """Save the survival analysis model and autoencoder to disk.

        Keyword Arguments:
        model_dir -- str, the directory where the models are saved (default "./models")
        """
        joblib.dump(self.model, os.path.join(model_dir, "survival.pkl"))
        joblib.dump(self.explainer, os.path.join(model_dir, "explainer.pkl"))
        self.autoencoder.save_model(model_dir)

    def run(self):
        """Run the survival analysis and print the concordance index for both original and compressed features."""
        self.autoencoder.fit()
        self.encoded_X_train = self.autoencoder.transform()
        self.model.fit(self.encoded_X_train, self.y)
        cindex = self.model.score(self.encoded_X_train, self.y)
        print(f"Concordance Index Compressed = {round(cindex, 3)}")
        self.model.fit(self.X, self.y)
        cindex = self.model.score(self.X, self.y)
        print(f"Concordance Index Original = {round(cindex, 3)}")

    def predict_survival_function(self, x_new, savefig=False):
        """Predict the survival function for new samples.

        Arguments:
        x_new -- pandas.DataFrame, the new samples to predict the survival function for

        Keyword Arguments:
        savefig -- bool, whether to save the plot of the survival functions to disk (default False)

        Returns: None
        """
        pred_surv = self.model.predict_survival_function(x_new)
        time_points = np.arange(125, 7185)
        for i, surv_func in enumerate(pred_surv):
            plt.step(
                time_points,
                surv_func(time_points),
                where="post",
                label="Sample %d" % (i + 1),
            )
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("time $t$ in days")
        plt.legend(loc="best")
        if savefig:
            plt.savefig(os.path.join("outputs", "survival.png"))

    def plot_important_features(self, top_n=20):
        """Plots the feature importances of the fitted machine learning model.

        Arguments:
            top_n (int): The number of top features to include in the plot. Defaults to 20.

        Returns: None.
        """
        feature_importance = self.model.feature_importances_
        feature_names = list(self.X_train.columns)
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        plt.figure(figsize=(14, 20))
        plt.barh(pos, feature_importance[sorted_idx], align="center")
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.title("Feature Importance")

    def plot_shap_waterfall(self, x_new, savefig=False):
        """Plot a SHAP waterfall plot for the new sample.

        Arguments:
        x_new -- pandas.DataFrame, the new sample to plot the SHAP waterfall for

        Keyword Arguments:
        savefig -- bool, whether to save the SHAP waterfall plot to disk (default False)

        Returns: None
        """
        shaps = self.explainer(x_new)
        shap.plots.waterfall(shaps[0], max_display=20, show=False)
        if savefig:
            plt.savefig(os.path.join("outputs", "shap_waterfall.png"))


if __name__ == "__main__":
    data_x, data_y = load_breast_cancer()
    X = OneHotEncoder().fit_transform(data_x)

    analysis = SurvivalAnalysis(X, data_y)
    analysis.run()
    analysis.predict_survival_function(X.iloc[0:1, :])
    analysis.plot_shap_waterfall(X.iloc[0:1, :])

    analysis.save_model()
