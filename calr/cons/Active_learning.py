import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
from .AdditionalFunction import AdditionalFunction


class Active_learning:
    def __init__(self, data, dataframe):
        self.__dataframe=dataframe
        self.__data=data
        self.__eps = 0.1  # config['eps']
        self.__min_pts = 20
        self.__core_point=[]
        self.__border_point=[]
        self.__additionalfunc=AdditionalFunction()
    def comput_core_border(self):
        for point in range(len(self.__data)):
            neighbour_s = self.__additionalfunc.region_query(point, self.__data, self.__eps)
            if len(neighbour_s) >= self.__min_pts:
                self.__core_point.append(point)
            else:
                self.__border_point.append(point)

    def active_selecting(self, must_link, cannot_link ):
        self.comput_core_border()
        selected = []
        a=len(must_link)
        b=len(cannot_link)
        querynum=a+b
        while True:
            if not selected:
                x=random.choice(self.__core_point)
                selected.append(x)
                querynum +=1
            else:
                x=self.get_farest_point_set(self.__core_point, selected)
                selected.append(x)    #not sure
                for y in selected:
                    self.addlabel([x,y],must_link, cannot_link)
                    querynum+=1
            b1 = self.get_nearest_point(x, self.__border_point)
            self.addlabel([x,b1],must_link,cannot_link)
            querynum += 1
            b2 = self.get_farest_point(x, self.__border_point)
            self.addlabel([x,b2],must_link,cannot_link)
            querynum += 1
            if a==len(must_link) and b==len(cannot_link):
                break
        print("constrainnum:",querynum)

    def addlabel(self, pair, must_link, cannot_link):
        if self.__dataframe.label[pair[0]] == self.__dataframe.label[pair[1]]:
            must_link.append(pair)
            self.__additionalfunc.remove_dupli(must_link)
        else:
            cannot_link.append(pair)
            self.__additionalfunc.remove_dupli(cannot_link)

    def get_nearest_point(self, point, pointset):
        min = self.__additionalfunc.get_distance(0, point, self.__data)
        result_index = pointset[0]
        for p in pointset:
            dist = self.__additionalfunc.get_distance(p, point, self.__data)
            if dist < min:
                min = dist
                result_index = p
        return result_index

    def get_farest_point(self, point, pointset):
        max = 0
        result_index = pointset[0]
        for p in pointset:
            dist = self.__additionalfunc.get_distance(p, point, self.__data)

            if dist > max:
                max = dist
                result_index = p

        return result_index

    def get_farest_point_set(self, pointset, scs):
        max = 0
        for p in pointset:
            dist_idx=self.get_nearest_point(p,scs)
            dist = self.__additionalfunc.get_distance(dist_idx, p, self.__data)
            if dist>max:
                max=dist
                result_index = p
        return result_index