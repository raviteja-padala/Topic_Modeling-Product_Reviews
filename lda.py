# -*- coding: utf-8 -*-
"""LDA_Topic_modelling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Uo-Wd1jrFSJkIWc2dXnytY83zm2a0Tr3
"""

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
import pandas as pd
import re

import emoji
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from nltk.corpus import stopwords

#pip install emoji


supplements_df = pd.read_csv("https://raw.githubusercontent.com/raviteja-padala/Datasets/main/supplements.csv")

df = supplements_df.copy()


# Remove "READ MORE" from the 'cleaned_comments' column
df['comment'] = df['comment'].str.replace('READ MORE', '')

"""# Data reprocessing

We have emojis in the comment data, have to handle the data with emojis

#### Rows with emojis
"""
# Regular expression to match emojis
emoji_pattern = re.compile("[\U00010000-\U0010ffff\uD800-\uDBFF\uDC00-\uDFFF]+", flags=re.UNICODE)

# Find rows with emojis in the 'comments' column
emoji_rows = df[df['comment'].str.contains(emoji_pattern)]

# Display rows with emojis
emoji_rows


# Load stop words once
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Lowercasing and splitting into words
    words = text.lower().split()

    # Removing Punctuation
    words = [word.strip(string.punctuation) for word in words]

    # Removing Stop words
    words = [word for word in words if word not in stop_words]

    # Tokenization
    tokens = words

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Rejoin tokens into a cleaned sentence
    cleaned_text = ' '.join(tokens)

    return cleaned_text

df['cleaned_comments'] = df['comment'].apply(preprocess_text)



"""# LDA modelling"""


from gensim.corpora import Dictionary, MmCorpus
from gensim.models.coherencemodel import CoherenceModel


# ... (Previous code)

if __name__ == '__main__':
    # Create a dictionary and document-term matrix
    dictionary = corpora.Dictionary(df[df['product_name'] == 'MUSCLEBLAZE Creatine']['cleaned_comments'].apply(str.split))
    corpus = [dictionary.doc2bow(comment.split()) for comment in df[df['product_name'] == 'MUSCLEBLAZE Creatine']['cleaned_comments']]

    # Train the LDA model
    num_topics = 4  # Specify the number of topics
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Print the topics and top words
    for topic_id, topic_words in lda_model.print_topics():
        print(f"Topic {topic_id}: {topic_words}\n")

    # Calculate Perplexity Score
    perplexity_score = lda_model.log_perplexity(corpus)
    print(f"Perplexity Score: {perplexity_score}")

    # Calculate Coherence Score
    coherence_model = CoherenceModel(model=lda_model, texts=df[df['product_name'] == 'MUSCLEBLAZE Creatine']['cleaned_comments'].apply(str.split), dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"Basic LDA Model Coherence Score: {coherence_score}")


    # Calculate document-topic distribution
    document_topic_distribution = [lda_model.get_document_topics(doc) for doc in corpus]

        # Calculate topic proportions
    topic_proportions = [dict(doc) for doc in document_topic_distribution]

        # Count the number of documents associated with each topic
    topic_counts = {topic_id: sum(1 for doc in topic_proportions if topic_id in doc) for topic_id in range(lda_model.num_topics)}

        # Calculate topic weights (proportion of documents)
    total_documents = len(corpus)
    topic_weights = {topic_id: count / total_documents for topic_id, count in topic_counts.items()}

    print(f"LDA Topic weights: {topic_weights}")


    

 # Extract significant keywords for each topic
    top_keywords_per_topic = []
    num_top_keywords = 5  # You can adjust this based on your preference

    for topic in lda_model.print_topics():
        top_keywords = re.findall(r'\"(.*?)\"', topic[1])[:num_top_keywords]
        top_keywords_per_topic.append(top_keywords)


        # Define rules to map keywords to labels (customize as needed)
        label_rules = {
            "Taste ,  Quality, and Ingredients": ["taste", "weight", "protein", "month", "like", "milk"],
            "Positive Product Attributes and Quality": ["product", "nice", "original", "best", "effect", "genuine", "muscle"],
            "Positive Experiences and Results": ["good", "take", "work", "body"],
            "Daily Impact, Mixability, and Workout Effects": ["creatine", "water",  "result", "workout", "drink"],
            "Negative Feedback and Quality": ["bad", "packing", "worst", "test", "horrible", "super"],
            "Money, Delivery, Feedback": ["money", "time", "waste", "delivery"],
            "Positive Feedback and Weight Gain": ["gain", "mass", "kg"],
            "Positive Feedback and Quality": ["genuine"],
            "Product Usage ": ["used", "use"],
            "Product Quality and Satisfaction": ["excellent", "really", "thanks"],
            "Fish Oil and Recommendation": ["fish", "oil", "better"],
            "Product Feedback and Effectiveness": ["effect"],
            "Health and Quality": ["health"]
        }

        # ...

# Automatically generate labels for each topic based on extracted keywords
    topic_labels = []
    unique_labels = set()  # Create a set to store unique labels

    for keywords in top_keywords_per_topic:
        matched_labels = []

    for label, rule_keywords in label_rules.items():
        if any(keyword in rule_keywords for keyword in keywords):
            matched_labels.append(label)

    if matched_labels:
        unique_matched_labels = ", ".join(matched_labels)
        # Check if the label is already in the set before adding it
        if unique_matched_labels not in unique_labels:
            topic_labels.append(unique_matched_labels)
            unique_labels.add(unique_matched_labels)

            # Merge all unique labels into a single string
            unique_topic_labels = ", ".join(unique_labels)

    print(f"Unique Topic Labels: {unique_topic_labels}")

    print('-'*50)

    print(f"Topic Labels: {topic_labels}")

    