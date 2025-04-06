from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

print(word_tokenize("Привет, как дела?", language="russian"))