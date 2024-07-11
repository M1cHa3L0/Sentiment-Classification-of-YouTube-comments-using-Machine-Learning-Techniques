import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')


# Initialize the stemmer and stop words list
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Path to your file
file_path = '/Users/apple/Desktop/final project/FinalProject/temporary YouTube queriesVideosTM.comments.txt'

# Read the file into a DataFrame
df = pd.read_csv(file_path, delimiter='\t')

# Define a function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words and apply stemming
    processed_tokens = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words]
    # Join tokens back into a single string
    return ' '.join(processed_tokens)

# Apply the preprocessing function to the 'CommentTextDisplay' column
df['ProcessedComment'] = df['CommentTextDisplay'].apply(preprocess_text)

# Display the first few entries to verify
print(df[['CommentTextDisplay', 'ProcessedComment']].head())
