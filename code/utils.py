import torch
import torch.nn as nn
import numpy as np
import glob

#Gives all the words to its glove embedding matrix (dictionary contains 0 as'<PAD>')
def process_dict(glove_path, dim):
    with open(glove_path, 'rb') as f:
        glove_dict = {'<PAD>' : [0]*dim}
        for l in f:
            line = l.decode('utf-8').split()
            word = line[0]
            glove_dict[word] = line[1:]
        return glove_dict

def getEmbed_layer(word2idx, glove_dict, dim):
    matrix_len = len(word2idx)

    #shape : (total vocab size)xdim
    weights_matrix = np.random.rand(matrix_len, dim)
    words_not_found = []

    for word,i in word2idx.items():
        if word in glove_dict:
            weights_matrix[i] = glove_dict[word]
        else:
            words_not_found.append(word)

    print('Number of words not found in glove dictionary: {}'.format(len(words_not_found)))
    #print(words_not_found)
    #Make sure index at 0 corresponds to padding
    #assert (list(weights_matrix[1,:]) == [0]*dim)
     
    num_embeddings, embedding_dim = weights_matrix.shape

    #num_embeddings -> vocab_size
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
    emb_layer.weight.requires_grad = True
    return emb_layer

#get all the words present in all the description
def getVocab(dataset_path):
    all_files = glob.glob(dataset_path + '/**/**/*.txt')
    y = set()
    for file_ in all_files:
        with open(file_) as f:
            y = y.union(set(re.findall(r'[\w]+', f.read())))
    return list(y)

#Create mat file of incides for a list of class
class saveMat:
    def __init__(self,word2idx):
        self.word2idx= word2idx
    def get_desc(self,classList, subfolder): #classesList: classes['val'] or classes['test']
        ans = {} #mat file
        #subfolder = 'val' if valClasses else 'test'
        for class_ in classList:
            all_desc_forAClass = []
            getFiles = glob.glob('../text_c10/{}/{}/*.txt'.format(subfolder,class_), recursive=True)
            for file_ in getFiles:
                with open(file_) as f:
                    all_desc_forAClass += [self.getIndices_forALine(line) for line in f.readlines()]   
            ans[class_] = np.array(all_desc_forAClass)
        return ans

    def getIndices_forALine(self,desc):
        words = re.findall(r'[\w]+',desc)
        idx = np.zeros(30)
        i = 0
        for word in words:
            if word in self.word2idx:
                idx[i] = self.word2idx[word]
                i += 1
            if i==30:
                break
        return idx
    



if __name__=='__main__': pass
    #word2idx, glove_dict = process_dict('glove.6B.50d.txt')
    #layer = getEmbed_layer(word2idx=word2idx, glove_dict=glove_dict)
