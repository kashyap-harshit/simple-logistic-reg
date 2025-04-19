import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# nltk.download('stopwords')

df = pd.read_csv("sms.tsv", sep='\t', header=None, names=['label', 'message'])

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def cleanText(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words] #one cool function
    return ' '.join(words)
    
df["cleaned"] = df['message'].apply(cleanText)


vectorizer = CountVectorizer()
print(vectorizer.fit_transform(df['cleaned']).toarray())