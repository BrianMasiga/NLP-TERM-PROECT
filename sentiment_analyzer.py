# %%
import gradio as gr
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# %%
# Load the dataset
data = pd.read_csv("swahilidataset.csv")

# Print the first 10 rows of the dataset
print(data.head(10))

# %%

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets
print("Training set shape:", train_data.shape)
print("Testing set shape:", test_data.shape)

# %%

# Defining my custom list of swahili stop words
my_sw_stop_words = ["akasema", "alikuwa", "alisema", "baada", "basi", "bila", "cha", "chini", "hadi", "hapo", "hata", "hivyo", "hiyo", "huku", "huo", "ili", "ilikuwa", "juu", "kama", "karibu", "katika", "kila", "kima", "kisha", "kubwa", "kutoka", "kuwa", "kwa", "kwamba", "kwenda", "kwenye", "la", "lakini", "mara", "mdogo",
                    "mimi", "mkubwa", "mmoja", "moja", "muda", "mwenye", "na", "naye", "ndani", "ng", "ni", "nini", "nonkungu", "pamoja", "pia", "sana", "sasa", "sauti", "tafadhali", "tena", "tu", "vile", "wa", "wakati", "wake", "walikuwa", "wao", "watu", "wengine", "wote", "ya", "yake", "yangu", "yao", "yeye", "yule", "za", "zaidi", "zake"]

# Create a CountVectorizer object with your custom stop words
vectorizer = CountVectorizer(stop_words=my_sw_stop_words)

# Vectorize the training data
train_vectors = vectorizer.fit_transform(train_data['text'])

# Vectorize the testing data
test_vectors = vectorizer.transform(test_data['text'])

# Train the Naïve Bayes model
nb_model = MultinomialNB()
nb_model.fit(train_vectors, train_data['sentiment'])

# Evaluate the Naïve Bayes model
nb_predictions = nb_model.predict(test_vectors)
print("Naïve Bayes accuracy:", accuracy_score(
    test_data['sentiment'], nb_predictions))
print("Naïve Bayes precision:", precision_score(
    test_data['sentiment'], nb_predictions, average='weighted'))
print("Naïve Bayes recall:", recall_score(
    test_data['sentiment'], nb_predictions, average='weighted'))
print("Naïve Bayes confusion matrix:\n", confusion_matrix(
    test_data['sentiment'], nb_predictions))

# Training the Support Vector Machine model with a higher max_iter parameter
svm_model = LinearSVC(max_iter=50000)
svm_model.fit(train_vectors, train_data['sentiment'])

# Evaluate the Support Vector Machine model
svm_predictions = svm_model.predict(test_vectors)
svm_accuracy = accuracy_score(test_data['sentiment'], svm_predictions)
svm_precision = precision_score(
    test_data['sentiment'], svm_predictions, average='weighted')
svm_recall = recall_score(
    test_data['sentiment'], svm_predictions, average='weighted')
svm_confusion_matrix = confusion_matrix(
    test_data['sentiment'], svm_predictions)

print("Support Vector Machine accuracy:", svm_accuracy)
print("Support Vector Machine precision:", svm_precision)
print("Support Vector Machine recall:", svm_recall)
print("Support Vector Machine confusion matrix:\n", svm_confusion_matrix)


# %%
# Load the trained Support Vector Machine model
svm_model = LinearSVC()
svm_model.fit(train_vectors, train_data['sentiment'])

# Process the new tweet
tweet = "wimbo mzuri sana unafundisha kwa hawa warembo wetu wa kileo majivuno kibao"
new_tweet_vector = vectorizer.transform([tweet])

# Predict the sentiment of the new tweet
new_tweet_sentiment = svm_model.predict(new_tweet_vector)

# Print the sentiment of the new tweet
print("The sentiment of this swahili text is : " + new_tweet_sentiment[0])

# %%


def predict_sentiment(tweet):
    # Process the new tweet
    new_tweet_vector = vectorizer.transform([tweet])

    # Predict the sentiment of the new tweet
    new_tweet_sentiment = svm_model.predict(new_tweet_vector)

    return new_tweet_sentiment[0]


tweet_input = gr.inputs.Textbox(lines=3, label="Enter your tweet")
prediction = gr.outputs.Label(label="The sentiment of this swahili text is : ")

gr.Interface(fn=predict_sentiment, inputs=tweet_input, outputs=prediction, title="Swahili Sentiment Analyzer by Brian Masiga 19S01ACS009",
             description="Enter a swahili text and get its sentiment prediction.").launch(share=True)
