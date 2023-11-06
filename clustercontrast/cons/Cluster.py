from .AdditionalFunction import AdditionalFunction
from collections import defaultdict


class Cluster(object):
    def __init__(self, name, clusterid):
        self.name = name
        self.points = []
        self.clusterid = clusterid
        self.label = 0
        self.ad=AdditionalFunction()

    def add_point(self, point):
        if isinstance(point, list):
            self.points += point
        else:
            self.points.append(point)

    def get_points(self):
        return self.points

    def delete_point(self,point):
        self.points.remove(point)

    # def obtain_label(self, dataframe):
    #     label_frequency=defaultdict()
    #     clusterdata = dataframe.iloc[self.points]
    #     for elemtent in clusterdata:
    #         label_frequency[elemtent.label]+=1
    #     return label_frequency

    def obtain_label(self, dataframe):
        clusterdata = dataframe.iloc[self.points]
        frequent_list = self.ad.most_common(clusterdata.label)
        return frequent_list

    def get_label(self):
        return self.label

    def erase(self):
        self.points = []
    def has(self, point):
        return point in self.points
    def get_clusterid(self):
        return self.clusterid
    def getname(self):
        return self.name
    def __str__(self):
        return "%s: %d points" % (self.name, len(self.points))

