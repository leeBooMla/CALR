from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk.stem import PorterStemmer
from sklearn.neighbors import NearestNeighbors

import random
import itertools
import numpy as np

class AdditionalFunction:
    def __init__(self):
        self.__ml=[]
        self.__cl=[]
        self.__data_per_label=dict()
        self.__neighborhood=np.array([])
    def get_mustlink(self):
        return self.__ml


    def region_query(self, point, data, eps):
        if self.__neighborhood.size == 0:
            neighbors_model = NearestNeighbors(radius=(1-eps), metric='cosine',
                     metric_params=None, algorithm='auto', leaf_size=30, p=-1)
            neighbors_model.fit(data)
            nbrs = neighbors_model.radius_neighbors(data, (1-eps), return_distance=False)
            self.__neighborhood = nbrs
        nbrs = self.__neighborhood[point].tolist()
        nbrs.remove(point)
        return nbrs


    def get_distance( self, from_point, to_point, data):  #result is a list, recalculate cosine according to the paper
        fro = data[from_point]
        to = data[to_point]
        return float(cosine_similarity([fro], [to]))  ##document should be in the form of [[],[]]

    def get_must_link(self, data, constrain_num):
        result=[]
        for item in set(data.label):
            l = data[data.label == item].index.values.tolist()
            m = list(itertools.combinations(l, 2))
            result.extend(m)
        '''for item in data.values():
            for key,value in data.items():
                if value == item :
                    l = key.tolist()
            m = list(itertools.combinations(l, 2))
            result.extend(m)'''
        self.__ml=result
        return random.sample(result,constrain_num)

    def increase_ml(self, ml, increase_num):
        self.__ml = [ elem for elem in self.__ml if elem  not in ml]
        if len(self.__ml)>increase_num:
            ml.extend(random.sample(self.__ml, increase_num))
        else:
            ml.extend(self.__ml)
        return ml


    def remove_dupli(self, listdata):  #for list of list like [[1,2],[2,3]]
        seen=set()
        newlist=[]
        for item in listdata:
            t = tuple(item)
            if t not in seen and t[0]!=t[1] and t[::-1] not in seen:
                newlist.append(item)
                seen.add(t)
        return newlist

    def get_connot_link(self, data, constrain_num):
        result=[]
        data_per_label = dict()
        for item in set(data.label):
            data_per_label[item] = data[data.label == item].index.values.tolist()
        if constrain_num > 0:
            for i in range(constrain_num):
                label1 = random.choice(list(data_per_label.keys()))
                label2 = random.choice(list(data_per_label.keys()))
                while label1 == label2:
                    label2 = random.choice(list(data_per_label.keys()))
                result.append([random.choice(data_per_label[label1]), random.choice(data_per_label[label2])])
        self.__data_per_label = data_per_label
        self.__cl = result
        return self.__cl

    def increase_cl(self, cl, increase_num):
        for i in range(increase_num):
            label1 = random.choice(list(self.__data_per_label.keys()))
            label2 = random.choice(list(self.__data_per_label.keys()))
            while label1 == label2:
                label2 = random.choice(list(self.__data_per_label.keys()))
            pair = [random.choice(self.__data_per_label[label1]), random.choice(self.__data_per_label[label2])]
            while pair in self.__cl:
                pair = [random.choice(self.__data_per_label[label1]), random.choice(self.__data_per_label[label2])]
            self.__cl.append(pair)
        return self.__cl

    def transitive_closure(self, elements):   #mistake , read on paper
        elements = [tuple(elem) for elem in elements]
        closure = set(elements)
        while True:
            new_relations = set((x,w) for x,y in closure for q,w in closure if q == y)
            closure_until_now = closure | new_relations
            if closure_until_now == closure:
                break
            closure = closure_until_now
        return list([k, i] for (k,i) in closure)


    #DG: addition - improved this function
    def update_cannotlink(self, CLlist, Tlist):
        flattened_cl = list(sum(CLlist, []))
        for tl in Tlist:
            if tl[0] in flattened_cl or tl[1] in flattened_cl:
                for cl in CLlist:
                    if tl[0] == cl[0] and [tl[1], cl[1]] not in CLlist:
                        CLlist.append([tl[1], cl[1]])
                    if tl[0] == cl[1] and [tl[1], cl[0]] not in CLlist:
                        CLlist.append([tl[1], cl[0]])
                    if tl[1] == cl[0] and [tl[0], cl[1]] not in CLlist:
                        CLlist.append([tl[0], cl[1]])
                    if tl[1] == cl[1] and [tl[0], cl[0]] not in CLlist:
                        CLlist.append([tl[0], cl[0]])
        return CLlist



    def common_member(self, a, b):
        a_set = set(a)
        b_set = set(b)
        if (a_set & b_set):
            return True
        else:
            return False

    def contain_relation(self, a, b):
        a_set = set(a)
        b_set = set(b)
        if a_set.issubset(b_set) or b_set.issubset(a_set):
            return True
        else:
            return False


    def most_common(self, lst):
        count = lst.value_counts()
        frequency_percentage=count.divide(count.sum())
        return frequency_percentage


    def get_documentpair(self, numbers):
        # Generate all possible non-repeating pairs
        pairs = list(itertools.combinations(numbers, 2))
        random.shuffle(pairs)
        return pairs

    '''def stemming_tokenizer(self, text):
        stemmer = PorterStemmer()
        return [stemmer.stem(w) for w in word_tokenize(text)]'''