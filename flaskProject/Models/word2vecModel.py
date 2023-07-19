from gensim.models import Word2Vec


class Word2VecModel:
    def __init__(self):
        self.model = Word2Vec.load("../ModelFiles/word2vec.model")
        print("Word2Vec is loaded")


w2v = Word2VecModel()
