# oncolyze_survival
Survival analysis for multi-modal data using deep learning, and a streamlit web app for user interaction

Use an encoder-decoder architecture to compress and consume multi-modal inputs to extract features from the encoder kernel.

Use survival analysis using gradient boosted trees with scikit-survival library, with features from encoder as well as clinical, genetic features.

Streamlit is used to build a web-app for interaction, training visualization and prediction (inference).

Initial version supports breast cancer. 

TODO:
Automatically bring in new datasets for new types of cancer. Minimal data engineering (only data preprocessing) needed.

Project Abstract:
Background:
Nearly one in eight women will develop breast cancer during her lifetime according to ACS, and total incidences are expected to climb to 3.2 million by 2050. Machine learning has become increasingly prevalent within oncological precision medicine. However, most modern algorithms either yield binary, unexplained results, predict only short-term survival, or require excessive memory. Survival analysis also presents a censorship issue, as continuous lifetime data cannot always be collected.

Approach:
The research objective is to create a Python web application that utilizes back-end deep learning for breast cancer survival analysis in remote regions. Input multi-omics data (multi-gene expression, estrogen receptor status, clinical data) is funneled through an autoencoder architecture to reduce feature dimensionality. A gradient-boosted Cox Regression model learns from the omics data, with Cox Partial Likelihood (CPL) as a loss and Concordance Index as the evaluation metric to optimize. The Kaplan-Meier survival function, from Python’s scikit-survival library, is then utilized to estimate survival probability as a function of time while accommodating right-censored data. 

Results/Discussion:
The final model attains a c-index of 86%. The algorithm is fully explainable, outputting Shapley values for each genetic risk factor and box plots to assess statistical significance. The trained model is offloaded to an interactive Plotly application that predicts survival based on inputted patient biomedical data. The completed tool enables clinicians to achieve on-demand, explainable diagnoses in remote areas without the burden of hefty computation. For future study, the model will be trained on X cancer datasets and accept additional modalities of omics data.
