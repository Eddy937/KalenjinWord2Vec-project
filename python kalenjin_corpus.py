import re
re
import re
from nltk.tokenize import wordpunct_tokenize
from gensim.models import Word2Vec
import nltk
kalenjin_sentences = [
    "Kiptaiyat ak muren eng kisumet.",
    "Koee inendet ab kasit ne kikoomi.",
    "Kimnai lagok chemi konom kongoi.",
    "Kapkutuny nebo boiyot komwa.",
    "Amun kalyet nebo kotik kiptendeny."
]
def preprocess_text(sentences):
    """
    Clean and tokenize sentences.
    """
    clean_sentences = []
    for sentence in sentences:

        sentence = re.sub(r"[^\w\s]", "", sentence).lower()

        tokens = wordpunct_tokenize(sentence)
        clean_sentences.append(tokens)
    return clean_sentences
preprocessed_corpus = preprocess_text(kalenjin_sentences)
print("Preprocessed Corpus:", preprocessed_corpus)

model = Word2Vec(
    sentences=preprocessed_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=2
)
model.save("kalenjin_word2vec.model")
print("Word2Vec model saved as 'kalenjin_word2vec.model'.")
# Test the Model
word_vector = model.wv['kiptaiyat']
print("\nVector for 'kiptaiyat':\n", word_vector)
similar_words = model.wv.most_similar('kiptaiyat')
print("\nWords similar to 'kiptaiyat':", similar_words)