# Import necessary libraries
from flask import Flask, request, render_template
import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
import requests
from bs4 import BeautifulSoup as bs
from logger import logging
from logger import logger
from exception import CustomException, handle_exception

# Create a Flask app
app = Flask(__name__)

# Define the dictionary to map product names to their corresponding URLs
product_urls = {
    "Endura Mass": "https://www.flipkart.com/endura-mass-weight-gainers-mass-gainers/product-reviews/itmechpae5uvhvy5?pid=PSLECHPASHZXHAKN&lid=LSTPSLECHPASHZXHAKN7QEWJF&marketplace=FLIPKART&page=",
    "BIGMUSCLES NUTRITION Premium Gold Whey": "https://www.flipkart.com/bigmuscles-nutrition-premium-gold-whey-protein/product-reviews/itm9818bbc1527a7?pid=PSLFV8R3VYCE6UQY&lid=LSTPSLFV8R3VYCE6UQYUAPQWZ&marketplace=FLIPKART&page=",
    "MUSCLEBLAZE Raw Whey Protein Concentrate": "https://www.flipkart.com/muscleblaze-raw-whey-protein-concentrate-80-digestive-enzymes-labdoor-usa-certified/product-reviews/itm003220323855d?pid=PSLET4NDF84FHZGV&lid=LSTPSLET4NDF84FHZGVGM5KKH&marketplace=FLIPKART&page=",
    "MUSCLEBLAZE Creatine Monohydrate": "https://www.flipkart.com/muscleblaze-creatine-monohydrate-india-s-only-labdoor-usa-certified/product-reviews/itm023e00803b96e?pid=PSLEFFGXKSA9QXRN&lid=LSTPSLEFFGXKSA9QXRN6NHKJJ&marketplace=FLIPKART&page=",
    "HEALTHKART HK Vitals Multivitamin": "https://www.flipkart.com/healthkart-hk-vitals-multivitamin-fish-oil-30n-tabs-30n-softgel-caps-2-piece-s-pack/product-reviews/itmede8b87172603?pid=VSLG6GZFHNWGY4KH&lid=LSTVSLG6GZFHNWGY4KHEQXADD&marketplace=FLIPKART&page="
}

# Define the step-by-step progress messages
progress_messages = [
    "Getting reviews",
    "Preprocessing text",
    "Modeling data",
    "Assigning Labels"
]

# Preprocess the data (reuse your preprocessing functions)
stop_words = set(stopwords.words('english'))



def preprocess_text(text):
    """
    Preprocesses text data by:
    - Lowercasing
    - Removing punctuation
    - Removing stopwords
    - Tokenization
    - Lemmatization

    Args:
        text (str): Input text to be preprocessed.

    Returns:
        str: Cleaned and preprocessed text.
    """
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

# Define your LDA model training function with exception handling
def train_lda_model(product_name):
    """
    Trains an LDA (Latent Dirichlet Allocation) model on product reviews to identify topics.

    Args:
        product_name (str): Name of the selected product.

    Returns:
        int: Total number of reviews.
        list: LDA model topics.
        dict: Topic weights.
        list: Topic labels.
    """
    try:
        # Extract the URL based on the selected product name
        product_url = product_urls.get(product_name)

        if not product_url:
            # Raise a custom exception for an invalid product name
            raise CustomException("Invalid product name")

        comments = []
        comment_headers = []
        ratings = []

        for page_num in range(1, 42):
            url = f"{product_url}{page_num}"

            response = requests.get(url)
            soup = bs(response.content, 'html.parser')
            all_rev = soup.find_all('div', {'class': "_3LWZlK _1BLPMq"})

            for j in range(0, 10):
                try:
                    comment_header = soup.find_all(
                        'div', {'class': "col _2wzgFH K0kLPL"})[j].p.text
                except:
                    comment_header = "No header"

                comment_headers.append(comment_header)

                try:
                    rating = soup.find_all(
                        'div', {'class': "_3LWZlK _1BLPMq"})[j].text
                except:
                    rating = "No rating"

                ratings.append(rating)

                try:
                    comment = soup.find_all('div', {'class': "t-ZTKy"})[j].text
                except:
                    comment = 'No comment'

                comments.append(comment)

        data = {
            'product_name': product_name,
            'comment_header': comment_headers,
            'comment': comments,
            'rating': ratings
        }

        supplements_df = pd.DataFrame(data, index=range(1, len(ratings) + 1))

        # Preprocess the comments
        supplements_df['comment'] = supplements_df['comment'].str.replace(
            'READ MORE', '')
        supplements_df['cleaned_comments'] = supplements_df['comment'].apply(
            preprocess_text)

        # Create a dictionary and document-term matrix
        dictionary = corpora.Dictionary(
            supplements_df['cleaned_comments'].apply(str.split))
        corpus = [dictionary.doc2bow(comment.split())
                  for comment in supplements_df['cleaned_comments']]

        # Train the LDA model
        num_topics = 4  # Specify the number of topics
        lda_model = LdaModel(corpus, num_topics=num_topics,
                             id2word=dictionary, passes=10)

        # Calculate document-topic distribution
        document_topic_distribution = [
            lda_model.get_document_topics(doc) for doc in corpus]

        # Calculate topic proportions
        topic_proportions = [dict(doc) for doc in document_topic_distribution]

        # Count the number of documents associated with each topic
        topic_counts = {topic_id: sum(1 for doc in topic_proportions if topic_id in doc) for topic_id in
                        range(lda_model.num_topics)}

        # Calculate topic weights (proportion of documents)
        total_documents = len(corpus)
        topic_weights = {topic_id: round(count / total_documents, 3) for topic_id, count in
                         topic_counts.items()}
        
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
            else:
                topic_labels.append("Unlabeled")

        return len(supplements_df), lda_model.print_topics(), topic_weights, topic_labels

    except Exception as e:
        # Log and handle exceptions
        logger.exception("An error occurred in train_lda_model")
        raise CustomException("An error occurred while processing the request")


# Define a route for the homepage with the form
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Defines the homepage route for the Flask application.
    Handles user input for product selection, topic modeling, and result presentation.

    Returns:
        HTML template: Renders the HTML template with results.
    """
    if request.method == 'POST':
        product_name = request.form['product_name']
        try:
            num_reviews, topics, topic_weights, topic_labels = train_lda_model(
                product_name)
            return render_template('index.html', num_reviews=num_reviews, product_name=product_name, topics=topics,
                                   topic_weights=topic_weights, topic_labels=topic_labels)
        except CustomException as ce:
            # Handle custom exceptions
            return handle_exception(ce)
        except Exception as e:
            # Handle other exceptions
            return handle_exception(e)

    return render_template('index.html', num_reviews=None, product_name=None, topics=None, topic_weights=None, topic_labels=None)


if __name__ == '__main__':
    app.run(debug=True)

#if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=80, debug=True)
