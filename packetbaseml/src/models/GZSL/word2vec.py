import string
from gensim.models.word2vec import Word2Vec
from src.data.labels import labels


def create_gensim_model(word_vector_size: int) -> Word2Vec:
    corpus = []
    for label in labels:
        words = [
            word.translate(str.maketrans("", "", string.punctuation))
            for word in label.split()
        ]
        corpus.append(words)

    gensim_model = Word2Vec(corpus, vector_size=word_vector_size, min_count=1)
    return gensim_model


class LabelsEmbeddings:
    def __init__(self, word_vector_size: int):
        self.word_vector_size: int = word_vector_size
        self.model: Word2Vec = create_gensim_model(word_vector_size)

    def fix_vectors_sizes(self, vectors: list) -> list:
        fixed_vectors = []
        max_size = max([len(v) for v in vectors])
        for vector in vectors:
            size_diff = max_size - len(vector)
            vector.extend([[0] * self.word_vector_size] * size_diff)
            fixed_vectors.append(vector)
        return fixed_vectors

    def generate_vectors(self, labels: dict):
        vectors = []
        for label in labels:
            description_vector = []
            words = [
                word.translate(str.maketrans("", "", string.punctuation))
                for word in label.split()
            ]
            for word in words:
                if word in self.model.wv.index_to_key:
                    description_vector.append(self.model.wv[word])
                else:
                    description_vector.append(self.word_vector_size * [0])
            vectors.append(description_vector)
        return self.fix_vectors_sizes(vectors)
