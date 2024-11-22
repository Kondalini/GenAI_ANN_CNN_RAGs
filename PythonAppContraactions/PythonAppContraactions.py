import re
import contractions
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

txt = "I can't believe it's already raining"
expanded_text= contractions.fix(txt)
print(expanded_text)

def expand_contractions(text):
    contractions_patern = {
        r"(?i) can't": "cannot",
        r"(?i) won't": "will not",
        r"(?i) it's": "is not" ,
        r"(?i) weren't": "where not" ,
        r"(?i) I'm": "I am" 
    }
    for contractions, expansion in contractions_patern.items():
        text = re.sub(contractions,expansion,text)

        return text

        txt = "I can't believe it's already raining and it's cold"
        expanded_text = expand_contractions(txt)
print(expanded_text)


nltk.download('punkt',download_dir= 'C:\\Users\\akrastev\\AppData\\Roaming\\nltk_data')
print(nltk.data.path)
txt = 'NLTK provides powerful tools for tokanization.It includes word tokanization and sentence tokanization'
words = word_tokenize(txt)
print(words)


        


