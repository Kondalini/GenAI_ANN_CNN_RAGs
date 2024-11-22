import nltk
from nltk import word_tokenize
from nltk import ngrams

nltk.download('punkt')

def generate_ngrams(text,n):
    tokens = word_tokenize(text)
    n_grams = list(ngrams(tokens,n))
    return n_grams



txt = "N-grams are sequence of n items from a give sample of text or speach"
unigrams = generate_ngrams(txt,1)
bigrams = generate_ngrams(txt,2)
trigrams = generate_ngrams(txt,3)