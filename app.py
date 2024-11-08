from flask import Flask, request, render_template
from flasgger import Swagger
import pickle
import re
import string
from nltk.corpus import stopwords

# Initialize the Flask app
app = Flask(__name__)
swagger = Swagger(app, template_file='swagger.yml')

# Load the trained model (make sure the path to your model is correct)
model = pickle.load(open('model_random_forest.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Function to clean user input
def clean_text(text):
    # Remove @username
    text = re.sub(r'@[^\s]+', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove emojis
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words
    text = ' '.join([word for word in text.split() if word.lower() not in stopwords.words('english')])
    return text.lower()

# Route for the HTML form and processing prediction
@app.route('/', methods=['GET', 'POST'])
def predict_sentiment():
    original_tweet = None
    get_sentiment = None
    
    if request.method == 'POST':
        # Get input text from user
        user_tweet = request.form['tweet']
        
        # Clean the text
        cleaned_tweet = clean_text(user_tweet)
        
        # Transform text to TF-IDF vector
        tweet_vector = tfidf.transform([cleaned_tweet])
        
        # Make prediction using the model
        prediction = model.predict(tweet_vector)
        
        # Map prediction to sentiment
        if prediction == 0:
            get_sentiment = 'Negative'
        elif prediction == 1:
            get_sentiment = 'Neutral'
        else:
            get_sentiment = 'Positive'
        
        # Set original tweet to display in the HTML
        original_tweet = user_tweet
    
    # Render the template with the sentiment result
    return render_template('index.html', original_tweet=original_tweet, get_sentiment=get_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
