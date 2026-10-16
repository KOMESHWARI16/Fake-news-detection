# Fake News Detection Using NLP

## 1. Project Overview

Fake News Detection Using Natural Language Processing (NLP) is a Python-based machine learning project that classifies news articles as "REAL" or "FAKE" based on their text.

The system accepts a news headline and article text from the user. It processes the text using TF-IDF vectorization and applies Logistic Regression to predict the news category. It also displays estimated prediction probabilities and graphical visualizations of the results.

This project demonstrates how NLP and machine learning can be applied to text classification.

## 2. Objectives

- To understand the fundamentals of Natural Language Processing.
- To convert text data into numerical features using TF-IDF.
- To train a Logistic Regression classification model.
- To predict whether entered news resembles REAL or FAKE training examples.
- To display prediction probabilities using a bar chart.
- To visualize the distribution of training articles using graphs.

## 3. Technologies Used

- Programming Language: Python
- Machine Learning: Scikit-learn
- NLP Technique: TF-IDF Vectorization
- Classification Algorithm: Logistic Regression
- Data Handling: Pandas
- Data Visualization: Matplotlib and Seaborn

## 4. Project Structure

text
Fake-News-Detection/
│
├── fake_news_detection.py
├── README.md
└── requirements.txt


## 5. Required Libraries

Install the required Python libraries using the following command:

bash
python -m pip install scikit-learn matplotlib seaborn pandas

Alternatively, create a `requirements.txt` file containing:

text
scikit-learn
matplotlib
seaborn
pandas


Install all dependencies with:

bash
python -m pip install -r requirements.txt


## 6. How the Project Works

### Step 1: Prepare the Training Data

A small collection of sample news articles is stored directly in the Python program. Each article is assigned a label: REAL or FAKE.

### Step 2: Convert Text into Numerical Features

The `TfidfVectorizer` converts news text into numerical features based on the importance of words in the training articles.

### Step 3: Train the Model

The Logistic Regression algorithm learns patterns from the vectorized articles and their labels.

### Step 4: Enter News Details

The user enters a news headline and the corresponding article text.

### Step 5: Predict the News Category

The trained model predicts whether the input resembles the REAL or FAKE examples it learned from.

### Step 6: Display Prediction Probabilities

The system displays the model's estimated probability for each category.

### Step 7: Generate Graphs

The program displays three graphs at the end:

1. Training Data Distribution: A bar chart comparing the number of REAL and FAKE training articles.
2. Prediction Probabilities: A bar chart showing the estimated probabilities for the entered news.
3. Word Contributions: A chart showing how words present in the input contribute to the model's prediction.

## 7. How to Run the Project

1. Install Python on your computer.
2. Open the project folder in Visual Studio Code.
3. Open the terminal in VS Code.
4. Install the required libraries.
5. Save the Python program as `fake_news_detection.py`.
6. Execute the following command:

bash
python fake_news_detection.py

7. Enter a news headline when prompted.
8. Enter the corresponding article text.
9. View the prediction, estimated probabilities, and graphs.

## 8. Sample Input

Headline:

text
Chandrayaan-3 successfully landed on the Moon

Article Text:

text
India's Chandrayaan-3 mission successfully landed near the Moon's south polar region on August 23, 2023.

## 9. Sample Output

text
===== FAKE NEWS DETECTION SYSTEM =====
Model training completed.

Enter the news headline: Chandrayaan-3 successfully landed on the Moon
Enter the news article text: India's Chandrayaan-3 mission successfully landed near the Moon's south polar region on August 23, 2023.

===== PREDICTION RESULT =====
Headline: Chandrayaan-3 successfully landed on the Moon
Prediction: REAL

===== PREDICTION PROBABILITIES =====
FAKE: [model-generated percentage]%
REAL: [model-generated percentage]%

Note: The prediction and percentages shown above are illustrative. Actual output depends on the trained model and input text.*

## 10. Advantages

- Easy to understand and implement.
- Does not require a separate CSV dataset for the initial demonstration.
- Demonstrates text preprocessing and machine learning classification.
- Provides graphical representations of prediction probabilities.
- Can be extended to work with larger news datasets.

## 11. Limitations

- The initial model uses only a small number of sample articles.
- Predictions may be incorrect for unfamiliar news topics.
- Estimated probabilities do not guarantee that an article is true or false.
- The current system does not verify news against trusted external sources.
- Reliable performance requires a larger, diverse, correctly labelled dataset and evaluation on unseen articles.

## 12. Future Enhancements

- Train the model using a larger, publicly available news dataset.
- Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Add a graphical user interface for easier interaction.
- Introduce more advanced NLP models.
- Integrate trusted news sources and fact-checking services.
- Calibrate and evaluate prediction probabilities.

## 13. Conclusion

The Fake News Detection Using NLP project demonstrates how Natural Language Processing and machine learning can classify text into predefined categories. TF-IDF extracts numerical text features, while Logistic Regression predicts a news label. Graphical visualizations help users understand the model's output.

This project provides a foundation for learning text classification. Its predictions should be treated as preliminary estimates rather than verified judgments of factual accuracy.

