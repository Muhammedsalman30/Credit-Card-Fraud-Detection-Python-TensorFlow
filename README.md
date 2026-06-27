# Credit Card Fraud Detection using Deep Autoencoder
Built a Deep Autoencoder model to detect fraudulent credit card transactions using unsupervised anomaly detection — tackling one of the most critical challenges in fintech: class imbalance.
Approach: Instead of traditional classification, the autoencoder was trained exclusively on normal transactions to learn their reconstruction patterns. Fraudulent transactions, being anomalies, produce significantly higher reconstruction errors — which are used as the detection signal.

## Key highlights:

Preprocessed 10,000 transactions — handled categorical encoding, feature scaling with StandardScaler, and removed noise columns
Designed a deep Encoder-Decoder architecture using Keras/TensorFlow, optimized with Adam and MSE loss
Applied threshold-based classification on reconstruction error to flag fraud
Evaluated with Classification Report, Confusion Matrix, and Reconstruction Error Distribution plots

Tech Stack: Python · TensorFlow/Keras · Scikit-learn · Pandas · Matplotlib 
