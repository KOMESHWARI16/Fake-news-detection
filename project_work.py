
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 1: Store sample news articles
articles = [
    "The government announced a new education policy today.",
    "The national cricket team won the championship.",
    "Scientists published a study on climate change.",
    "The central bank released its annual financial report.",
    "The space agency successfully launched a satellite.",
    "A new hospital was opened by the health department.",

    "Aliens landed in a city and took control of the government.",
    "Drinking a magic liquid makes humans live forever.",
    "A secret machine can make everyone instantly rich.",
    "Scientists confirmed that the moon is made of cheese.",
    "A person claims to have stopped the sun using a phone.",
    "Eating one special fruit gives humans unlimited powers."
]

# Step 2: Labels for the articles
labels = [
    "REAL", "REAL", "REAL",
    "REAL", "REAL", "REAL",
    "FAKE", "FAKE", "FAKE",
    "FAKE", "FAKE", "FAKE"
]

# Step 3: Create the NLP and ML pipeline
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)

# Step 4: Train the model
model.fit(articles, labels)

print("===== FAKE NEWS DETECTION SYSTEM =====")
print("Model training completed.")

# Step 5: Get user input
headline = input("\nEnter the news headline: ")
article_text = input("Enter the news article text: ")

news = headline + " " + article_text

# Step 6: Predict the category
prediction = model.predict([news])[0]
probabilities = model.predict_proba([news])[0]
classes = model.named_steps["logisticregression"].classes_

print("\n===== PREDICTION RESULT =====")
print("Headline:", headline)
print("Prediction:", prediction)

print("\n===== PREDICTION PROBABILITIES =====")
for category, probability in zip(classes, probabilities):
    print(f"{category}: {probability * 100:.2f}%")

# ==================================================
# ALL GRAPHS ARE DISPLAYED AT THE END
# ==================================================

sns.set_theme(style="whitegrid")

# Graph 1: Number of REAL and FAKE training articles
label_counts = pd.Series(labels).value_counts().reindex(
    ["REAL", "FAKE"], fill_value=0
)

plt.figure(figsize=(7, 5))
sns.barplot(
    x=label_counts.index,
    y=label_counts.values
)
plt.title("Training Data: REAL vs FAKE Articles")
plt.xlabel("News Category")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.show()

# Graph 2: Probabilities for the entered news
plt.figure(figsize=(7, 5))
probability_percentages = probabilities * 100

sns.barplot(
    x=list(classes),
    y=probability_percentages
)
plt.title(f"Prediction Probabilities: {prediction}")
plt.xlabel("Predicted Category")
plt.ylabel("Model Probability (%)")
plt.ylim(0, 100)

for index, value in enumerate(probability_percentages):
    plt.text(index, value + 1, f"{value:.1f}%", ha="center")

plt.tight_layout()
plt.show()

# Graph 3: Words with the strongest contribution to this prediction
vectorizer = model.named_steps["tfidfvectorizer"]
classifier = model.named_steps["logisticregression"]

# Transform the entered news into TF-IDF features
news_vector = vectorizer.transform([news])
feature_names = vectorizer.get_feature_names_out()

# For binary classification, positive coefficients favor classes_[1]
coefficients = classifier.coef_[0]
word_values = news_vector.toarray()[0]

# Calculate each present word's signed contribution
contributions = word_values * coefficients

# Reverse contribution direction if the predicted class is classes_[0]
if prediction == classes[0]:
    contributions = -contributions

# Keep only words that actually appear in the entered news
present_indices = [
    i for i, value in enumerate(word_values) if value > 0
]

# Sort by absolute contribution and select the top 10
top_indices = sorted(
    present_indices,
    key=lambda i: abs(contributions[i]),
    reverse=True
)[:10]


print("\n===== ANALYSIS COMPLETED =====")
