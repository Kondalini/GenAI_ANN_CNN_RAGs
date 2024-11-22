import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import string

# The text data
text = """Data plays a vital role in our everyday life.
Directly or indirectly, for daily life decisions, we depend on some data, be it choosing a novel to read from a list of books, buying a thing after considering the budget, and so on. Have you ever imagined searching for something on Google or Yahoo generates a lot of data?
This data is essential to analyze user experiences. Getting recommendations on various e-commerce websites after buying a product and tracking parcels during delivery are part of Data Analytics which involves analyzing the raw data to make informed decisions.
But this raw data does not help make decisions if it has some redundancy, inconsistency, or inaccuracy.
Therefore, this data needs to be cleaned before considering for analysis."""

# Preprocess the text: remove punctuation and convert to lowercase
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    return text

# Preprocess the original text
processed_text = preprocess_text(text)

# Split the text into sentences (or paragraphs)
sentences = processed_text.split('\n')

# Initialize the tokenizer
tokenizer = Tokenizer()

# Fit the tokenizer on the sentences
tokenizer.fit_on_texts(sentences)

# Print the number of unique words and word index
print(f"Number of unique words: {len(tokenizer.word_index)}")
print("Word Index:", tokenizer.word_index)

# Tokenize each sentence and print the result

input_sequences = []
for sentence in sentences:
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1,len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:i+1])
        print(input_sequences)
max_len = max([len(x) for x in input_sequences])

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')

X = padded_input_sequences[:,:-1]
y = padded_input_sequences[:,-1]
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=88)
y.shape
X.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
model = Sequential()
model.add(Embedding(88,100,input_length=33))
model.add(GRU(150))
model.add(Dense(88,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X,y,epochs=100)

text2 = "Data"

# tokenization
token_text = tokenizer.texts_to_sequences([text2])[0]
# padding
padded_text = pad_sequences([token_text], maxlen=33, padding='pre')
# model prediction
model.predict(padded_text)
text3 = "Data plays a vital"

# tokenization
token_text3 = tokenizer.texts_to_sequences([text3])[0]
# padding
padded_text3 = pad_sequences([token_text3], maxlen=33, padding='pre')
# model prediction
model.predict(padded_text3)

pos3 = np.argmax(model.predict(padded_text3))

for word, index in tokenizer.word_index.items():
  if index==pos3:
    print(word)
