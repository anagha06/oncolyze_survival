import matplotlib.pyplot as plt
import streamlit as st
from sksurv.datasets import load_breast_cancer
from sksurv.preprocessing import OneHotEncoder

from survival import SurvivalAnalysis


class StreamlitSurvivalDisplay:
    def __init__(self, survival_object, X_test):
        st.set_option("deprecation.showPyplotGlobalUse", False)
        self.survival_object = survival_object
        self.survival_model = survival_object.model
        self.X_test = X_test

        self.patient_id = 0
        self.cancer_type = "Breast"

    def st_title(self, text, color="green"):
        """Displays a centered, colored title on the Streamlit app.
        This is used instead of streamlit functions due to better formatting control.

        Args:
            text (str): The text to display in the title.
            color (str): The color of the title. Defaults to "green".

        Returns: None.
        """
        st.markdown(
            f"<h1 style='text-align: center; color: {color};'>{text}</h1>",
            unsafe_allow_html=True,
        )

    def st_subtitle(self, text, color="black"):
        """Displays a centered, colored subtitle on the Streamlit app.
        This is used instead of streamlit functions due to better formatting control.

        Args:
            text (str): The text to display in the title.
            color (str): The color of the title. Defaults to "green".

        Returns: None.
        """
        st.markdown(
            f"<h3 style='text-align: center; color: {color};'>{text}</h1>",
            unsafe_allow_html=True,
        )

    def display(self):
        """
        Main display for the app.
        """
        # Configuration.
        st.set_page_config(
            page_title="Patient Survival Rates",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Header.
        self.st_title("Oncolyze Survival Analysis")

        # Sidebar.
        self.cancer_type = st.sidebar.selectbox("Type of cancer", ["Breast", "Lung"])

        # Training and inference knobs.
        with st.sidebar.form("Show training data"):
            self.train_submit_button = st.form_submit_button(
                "Calculate Training Results"
            )

        self.patient_id = st.sidebar.slider("Patient ID", 0, 100, 0)

        with st.sidebar.form("Submit"):
            self.inf_submit_button = st.form_submit_button(
                "Calculate Inference Results"
            )

        self.inf_age = st.sidebar.slider("Patient Age", 1, 100)
        self.inf_size = st.sidebar.slider("Tumor Size", 0.1, 5.0)
        self.inf_er = st.sidebar.selectbox(
            "Estrogen Recepter", ["Positive", "Negative"]
        )
        self.inf_grade = st.sidebar.selectbox(
            "Tumor Grade",
            ["poorly differentiated", "intermediate", "well differentiated"],
        )

        self.X202240_at = st.sidebar.number_input("Gene X202240_at")
        self.X203391_at = st.sidebar.number_input("Gene X203391_at")
        self.X203306_s_at = st.sidebar.number_input("Gene X203306_s_at")
        self.X219724_s_at = st.sidebar.number_input("Gene X219724_s_at")
        self.X204540_at = st.sidebar.number_input("Gene X204540_at")

        # Update x_new based on user inputs.
        x_new = self.X_test.iloc[self.patient_id : self.patient_id + 1].copy()
        if self.inf_submit_button:
            self._update_x_new(x_new)

        # Training metrics: show for training selection.
        if self.train_submit_button:
            self.st_subtitle("Training Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Enc-Dec\ntrain loss", 0.01)
            col2.metric("Concordance\nIndex", 0.85)
            col3.metric("Training data", 200)
            style_metric_cards()

        # Show results based on training or inference.
        if self.train_submit_button or self.inf_submit_button:
            self.display_survival_prediction(x_new)
            self.display_shap_waterfall(x_new)

        # Only show for training selection.
        if self.train_submit_button:
            self.display_feature_importance()
            st.subheader("Test patients table")
            st.dataframe(self.X_test.style.highlight_max(axis=0), height=200)

    def _update_x_new(self, x_new):
        """
        Updates x_new DataFrame based on user inputs.
        """
        # Update the gene expression values from inputs.
        x_new.loc[self.patient_id, "X202240_at"] = (
            self.X202240_at
            if self.X202240_at > 0
            else x_new.loc[self.patient_id, "X202240_at"]
        )
        x_new.loc[self.patient_id, "X203391_at"] = (
            self.X203391_at
            if self.X203391_at > 0
            else x_new.loc[self.patient_id, "X203391_at"]
        )
        x_new.loc[self.patient_id, "X203306_s_at"] = (
            self.X203306_s_at
            if self.X203306_s_at > 0
            else x_new.loc[self.patient_id, "X203306_s_at"]
        )
        x_new.loc[self.patient_id, "X219724_s_at"] = (
            self.X219724_s_at
            if self.X219724_s_at > 0
            else x_new.loc[self.patient_id, "X219724_s_at"]
        )
        x_new.loc[self.patient_id, "X204540_at"] = (
            self.X204540_at
            if self.X204540_at > 0
            else x_new.loc[self.patient_id, "X204540_at"]
        )

        # Update the clinical data values from inputs.
        x_new.loc[self.patient_id, "age"] = self.inf_age
        x_new.loc[self.patient_id, "size"] = self.inf_size
        x_new.loc[self.patient_id, "er=positive"] = float(self.inf_er == "Positive")
        x_new.loc[self.patient_id, "grade=intermediate"] = float(
            self.inf_grade == "intermediate"
        )
        x_new.loc[self.patient_id, "grade=poorly differentiated"] = float(
            self.inf_grade == "poorly differentiated"
        )

        return x_new

    def display_survival_prediction(self, x_new):
        self.st_subtitle("Estimated Survival Probability")
        self.survival_object.predict_survival_function(x_new)
        st.pyplot(bbox_inches="tight")
        plt.clf()

    def display_shap_waterfall(self, x_new):
        self.st_subtitle("Shapley Scores")
        self.survival_object.plot_shap_waterfall(x_new)
        st.pyplot(bbox_inches="tight")
        plt.clf()

    def display_feature_importance(self):
        self.st_subtitle("Training Feature Importances")
        self.survival_object.plot_important_features()
        st.pyplot(bbox_inches="tight")
        plt.clf()


def style_metric_cards(
    background_color: str = "#FFF",
    border_size_px: int = 1,
    border_color: str = "#CCC",
    border_radius_px: int = 5,
    border_left_color: str = "#9AD8E1",
    box_shadow: bool = True,
):
    box_shadow_str = (
        "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
        if box_shadow
        else "box-shadow: none !important;"
    )
    st.markdown(
        f"""
        <style>
            div[data-testid="metric-container"] {{
                background-color: {background_color};
                border: {border_size_px}px solid {border_color};
                padding: 5% 5% 5% 10%;
                border-radius: {border_radius_px}px;
                border-left: 0.5rem solid {border_left_color} !important;
                {box_shadow_str}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_breast_cancer_data_and_model():
    # data_x = pd.read_csv(os.path.join("datasets/breast/", "x.csv"))
    # data_y = pd.read_csv(os.path.join("datasets/breast/", "y.csv")).to_records(index=False)
    data_x, data_y = load_breast_cancer()
    X = OneHotEncoder().fit_transform(data_x)

    analysis = SurvivalAnalysis(X, data_y, load_model=True)
    return analysis, X


if __name__ == "__main__":
    breast_analysis, breast_X = load_breast_cancer_data_and_model()
    display = StreamlitSurvivalDisplay(breast_analysis, breast_X)
    display.display()
