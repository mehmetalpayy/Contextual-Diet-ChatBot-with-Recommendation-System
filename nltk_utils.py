import numpy as np
import nltk
#nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """ split sentence into array of words/tokens a token can be a word or punctuation character, or number """
    return nltk.word_tokenize(sentence)


def stem(word):
    """stemming = find the root form of the word; examples:
    words = ["Organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    if sentence[i] is available in words[i], then put one for bog. 
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

def bow_with_tfidf(tokenized_sentence, all_words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(vocabulary=all_words, lowercase=True)
    tfidf_vector = tfidf_vectorizer.fit_transform([' '.join(sentence_words)])
    
    # Convert sparse matrix to dense array
    tfidf_array = np.array(tfidf_vector.todense())[0]
    
    return tfidf_array
