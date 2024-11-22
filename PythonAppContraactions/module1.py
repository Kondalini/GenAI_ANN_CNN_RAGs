import nltk

# Remove the punkt directory from nltk_data folder manually or programmatically:
# nltk.data.find('tokenizers/punkt')  # This will check if it exists.
# Delete 'punkt' folder if it exists.

# Now download the punkt package again:
nltk.download('punkt', download_dir='C:\\Users\\akrastev\\nltk_data')

from nltk.tokenize import word_tokenize

txt = 'NLTK provides powerful tools for tokenization. It includes word tokenization and sentence tokenization.'
words = word_tokenize(txt)
print(words)
