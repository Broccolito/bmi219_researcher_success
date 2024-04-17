# Predicting Researcher Success with Deep Learning

Welcome to the GitHub repository for our project on Predicting Researcher Success using Deep Learning. This is a class project of BMI219. This project explores the application of deep learning techniques to predict various measures of researcher success, such as the h-index, total citations, and recent performance metrics (h-index and citations since 2018). Our model leverages a sequence of neural network layers to understand and predict these indices, aiming to provide insights that could assist academic institutions, funding bodies, and policymakers.

## Project Overview

Researcher success is a multifaceted metric typically evaluated by indices like h-index and total citations. These indices help in understanding the impact and influence of a researcher's work. Our project builds a deep learning model to predict these metrics based on a researcher's publication data. The model's performance is benchmarked against traditional linear regression approaches such as Random Forest and Lasso Regression.

### Model Architecture

The deep learning model is structured as follows:

```python
model = Sequential([
    vectorize_layer,
    Embedding(max_features + 1, 16),
    LSTM(32, return_sequences=True),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    GlobalAveragePooling1D(),
    Dropout(0.2),
    Dense(1)
])
```

This model incorporates layers like LSTM and Dense with dropout for regularization, designed to process and learn from complex patterns in the input data.

## Datasets

The dataset used for training comprises academic publications' metadata, including:
- Author information
- Publication year
- Citation counts
- Journal impact factor
- Co-author network data

Data preprocessing steps include vectorization of textual data, normalization of numerical inputs, and encoding categorical variables.

## Performance Comparison

We compared our deep learning model against the following traditional regression methods:
- **Random Forest Regression**: A robust ensemble technique that uses multiple decision trees.
- **Lasso Regression**: A type of linear regression that applies a regularization penalty to the model coefficients.

Performance metrics such as Mean Squared Error (MSE), R² score, and Mean Absolute Error (MAE) were used to evaluate and compare the models.

## Installation

To set up the project environment:

```bash
git clone https://github.com/Broccolito/bmi219_researcher_success.git
cd bmi219_researcher_success
```

## Contributing

We welcome contributions to this project! If you have suggestions or improvements, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



