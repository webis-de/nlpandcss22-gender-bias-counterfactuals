from gensim.scripts.glove2word2vec import glove2word2vec

if __name__ == '__main__':
    glove2word2vec(glove_input_file="./glove.42B.300d/glove.42B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")