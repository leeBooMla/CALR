from sklearn.metrics import f1_score
from sqlalchemy import false
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, contingency_matrix
from sklearn.metrics import fowlkes_mallows_score, calinski_harabasz_score, f1_score, homogeneity_completeness_v_measure
import scipy.sparse as sp


from .Cluster import Cluster
from .AdditionalFunction import AdditionalFunction

class DBScanner:
    def __init__(self, data, dataframe, eps, min_pts):
        self.__eps = eps # config['eps']
        self.__min_pts = min_pts  # config['min_pts']
        self.__clusters = set()
        self.__cluster_count = 0
        self.__classified = []
        self.__TCS = []
        self.__data = data
        self.__df = AdditionalFunction()
        self.__dataframe=dataframe


    def printcluster(self):
        for element in self.__clusters:
            print("Show ",element.getname(),element.get_points())

    def dbscan(self, must_link, cannot_link):
        TCS = self.__df.transitive_closure(must_link)
        cannot_link = self.__df.update_cannotlink(cannot_link, TCS)
        name = 'cluster-%d' % self.__cluster_count
        new_cluster = Cluster(name, self.__cluster_count)

        # default noise cluster
        noise = Cluster('Noise', -1)
        #self.__clusters.add(noise)
        for point in range(self.__data.size(0)):
            if point not in self.__classified:
                if self.expand_cluster(new_cluster, point, cannot_link, TCS, noise):
                    self.__clusters.add(new_cluster)
                    self.__cluster_count += 1
                    name = 'cluster-%d' % self.__cluster_count
                    new_cluster = Cluster(name, self.__cluster_count)

        return self.__cluster_count

    def getlabel(self):
        assigned=set()
        clusterid=[]
        cluster_labels=pd.DataFrame({
            "clusterid":[],
            0:[],
            1:[],
            2:[],
        })
        for cluster in self.__clusters:
            frequency_labels=cluster.obtain_label(self.__dataframe)
            #cluster_labels=cluster_labels.append(frequency_labels)
            cluster_labels=pd.concat([cluster_labels, frequency_labels])
        cluster_labels.set_index('clusterid')
        cluster_labels.index=[ x.clusterid for x in self.__clusters]
        cluster_labels.fillna(value=0, inplace=True)

        #list(self.__clusters)
        if len(self.__clusters)>3:
            assigned_cluster = []
            copyed_cluster_labels = cluster_labels[:]
            for label in range(3):
                mostfrequent=copyed_cluster_labels[label].max()
                t=copyed_cluster_labels[copyed_cluster_labels[label]==mostfrequent]
                choosed_cluster=t.index[0]
                for cluster in self.__clusters:
                    if cluster.clusterid==choosed_cluster:
                        cluster.label=label
                copyed_cluster_labels=copyed_cluster_labels.drop(choosed_cluster)
                assigned_cluster.append(choosed_cluster)
            rest=self.__clusters-set(assigned_cluster)
            for cluster in rest:
                t=cluster.obtain_label(self.__dataframe).index[0]
                cluster.label=t
        #        cluster.label=cluster.obtain_label(self.__dataframe).index[0]
        else:
            assigned_label=[]
            for cluster in self.__clusters:
                frequency_list=cluster.obtain_label(self.__dataframe)
                if frequency_list.index[0] not in assigned_label:
                    cluster.label = frequency_list.index[0]
                    assigned_label.append(frequency_list.index[0])
                else:
                    if len(frequency_list)==1:
                        cluster.label = frequency_list.index[0]
                    if len(frequency_list) > 1 and frequency_list.index[1] not in assigned_label:
                        cluster.label = frequency_list.index[1]
                        assigned_label.append(frequency_list.index[1])
                    else:
                        if len(frequency_list)==3:
                            cluster.label = frequency_list.index[2]
                            assigned_label.append(frequency_list.index[2])



    def expand_cluster(self, cluster, point, cannot_link, TCS, noise):
        seed = []
        neighbour_pts = self.__df.region_query(point, self.__data, self.__eps)
        if len(neighbour_pts) < self.__min_pts:
            noise.add_point(point)
            self.__classified.append(point)
            return False
        inTCS=self.check_inlist(point, TCS)
        if inTCS:
            for pair in inTCS:
                for ele in pair:
                    if ele not in self.__classified:
                        cluster.add_point(ele)
                        self.__classified.append(ele)
                        seed.append(ele)
        else:
            cluster.add_point(point)
            self.__classified.append(point)
            seed.append(point)
        while (seed):
            first = seed[0]
            inTCS = self.check_inlist(first, TCS)
            if inTCS:
                for ele in inTCS:
                    for pt in ele:
                        if noise.has(pt) or pt not in self.__classified:
                            cluster.add_point(pt)
                            seed.append(pt)
                            self.__classified.append(pt)
                            if noise.has(pt):
                                noise.delete_point(pt)
            neighbour_s = self.__df.region_query(first, self.__data, self.__eps)
            if len(neighbour_s) >= self.__min_pts:
                if not self.__df.contain_relation(neighbour_s,seed):
                    for p in neighbour_s:
                        if (noise.has(p) or p not in self.__classified) and self.satisfy_cl(p, cluster, cannot_link):
                            cluster.add_point(p)
                            seed.append(p)
                            self.__classified.append(p)
                            if noise.has(p):
                                noise.delete_point(p)
            del seed[0]
        return True

    def satisfy_cl(self, p, cluster, cannot_link):
        if self.check_inlist(p, cannot_link):
            cluster_list = cluster.get_points()
            for item in cannot_link:
                if p in item:
                    s = [x for x in item if x != p]
                    if s[0] in cluster_list:
                        return False
        return True

    def check_inlist(self, point, list):
        result=[]
        for ele in list:
            if point in ele:
                result.append(ele)
        if(result):
            return result
        else:
            return False
    #
    # def average_test(self,dataframe):
    #     predict=[]
    #     truelabel=[]
    #     result=[]
    #     number=0
    #     for cluster in self.__clusters:
    #         if cluster.getname() != "Noise":
    #             indexlist=cluster.get_points()
    #             if len(indexlist) != 0:
    #                 clusterdata = dataframe.iloc[indexlist]
    #                 cpredict = [cluster.get_label()] * len(indexlist)
    #                 ctruelabel = list(clusterdata.label)
    #                 oneresult=normalized_mutual_info_score(cpredict, ctruelabel)
    #                 result.append(oneresult)
    #     print("Average NMI result:", sum(result) / len(result))

    def evaluation(self, dataframe):
        predict=[]
        truelabel=[]
        count=0
        for cluster in self.__clusters:
            if cluster.getname() != "Noise":
                indexlist=cluster.get_points()
                count+=len(indexlist)
                if len(indexlist) != 0:
                    clusterdata = dataframe.iloc[indexlist]
                    cpredict = [cluster.label] * len(indexlist)
                    ctruelabel = list(clusterdata.label)
                    predict.extend(cpredict)
                    truelabel.extend(ctruelabel)


        #print("predicted:",predict)
        #print("truelabel:",truelabel)
        result=normalized_mutual_info_score(truelabel, predict)
        fowlkes=fowlkes_mallows_score(truelabel, predict)
        hom = homogeneity_completeness_v_measure(truelabel, predict)
        fscore = f1_score(truelabel, predict, average='macro')
        pairwise_score = self.pair_wise_fmeasure(truelabel, predict)
        #return result, fowlkes, hom, fscore, pairwise_score
        return predict

    def pair_wise_fmeasure(self, labels_true, labels_pred):
        n_samples = len(labels_true)

        c = contingency_matrix(labels_true, labels_pred, sparse=True)
        c = c.astype(np.int64, copy= False)
        tp = np.dot(c.data, c.data) - n_samples
        fp = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
        fn = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        return 2 * precision * recall / (precision + recall)


    def f_measure(self, dataframe, ad):
        tpredict = 0
        fpositive = 0
        tscluster = len(ad._AdditionalFunction__ml)

        for cluster in self.__clusters:
            points = cluster.points
            pairs = self.__df.get_documentpair(points)
            for element in pairs:
                if self.check_cluster(element):
                    if self.check_label(element, ad):
                        tpredict += 1
                    else:
                        fpositive += 1
        precision = tpredict / (tpredict + fpositive)
        recall = tpredict / tscluster
        #print("precision:", precision)
        #print("recall:", recall)
        if precision == 0 or recall == 0:
            print("failed f_measure")
            return 0
        f_measure = 2 * precision * recall / (precision + recall)
        return f_measure

    #DG: addition - TP is the same as must link
    def check_label(self, pairs, ad):
        if pairs in ad._AdditionalFunction__ml:
            return True

    def check_cluster(self, pair):
        for cluster in self.__clusters:
            if pair[0] in cluster.get_points() and pair[1] in cluster.get_points():
                return True

