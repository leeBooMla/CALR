from cProfile import label
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#import logging
from DBScanner import DBScanner
from Active_learning import Active_learning
from AdditionalFunction import AdditionalFunction
import time
import matplotlib.pyplot as plt

#logging.basicConfig(format="%(asctime)s [%(name)s] [%(levelname)s]: %(message)s", level=logging.INFO)
#logger = logging.getLogger(__name__)
#logger.info("program started")


def main():
    dataframe = pd.read_csv("News-diff3.csv")
    label = dataframe.label
    text = dataframe.text
    ad=AdditionalFunction()
    ml=ad.get_must_link(dataframe,0)
    cl=ad.get_connot_link(dataframe,0)


    stemmed=[' '.join(ad.stemming_tokenizer(x)) for x in dataframe.text]
    vectorizer = TfidfVectorizer(stop_words="english")
    # With stemmming
    datavector = vectorizer.fit_transform(stemmed)
    #Without stemming
    #datavector = vectorizer.fit_transform(list(dataframe.text))


    # db= DBScanner(datavector,dataframe)
    # db.dbscan(ml, cl)
    # db.printcluster()
    # db.getlabel()
    # f_measure=db.f_measure(dataframe)
    # db.evaluation(dataframe)
    # db.average_test(dataframe)
    # print("y1:", f_measure)
    consnum=0
    fmeasure=[]
    conslist=[]
    Homogeneity=[]
    NMI=[]
    vmeasure=[]
    eps = 0.2
    min_pts = 10
    Completeness=[]
    Outfile = open("output_average_News-diff3.txt", "w")




    Outfile.write("Eps: %s, min_pts: % s \n" % (eps, min_pts))
    while consnum <= 400:
        average_nmi, average_fowlkes, average_hom, average_comp, average_v, average_fscore, average_pairwise_score = 0, 0, 0, 0, 0, 0, 0

        for elem in range(20):
            ml = ad.get_must_link(dataframe, 0)
            cl = ad.get_connot_link(dataframe, 0)
            if consnum > 0:
                ml = ad.increase_ml(ml, int(consnum / 2))
                cl = ad.increase_cl(cl, int(consnum / 2))
            db = DBScanner(datavector, dataframe, eps, min_pts)
            cluster_count = db.dbscan(ml, cl)
            print("Number of clusters found:", cluster_count)
            db.getlabel()
            print("constrain number:", consnum)
            nmi, fowlkes, hom, fscore, pairwise_score = db.evaluation(dataframe)
            average_nmi += nmi
            average_fowlkes += fowlkes
            average_hom += hom[0]
            average_comp += hom[1]
            average_v += hom[2]
            average_fscore += fscore
            average_pairwise_score += pairwise_score

        average_nmi = average_nmi / 20
        average_fowlkes = average_fowlkes / 20
        average_hom = average_hom / 20
        average_comp = average_comp / 20
        average_v = average_v / 20
        average_fscore = average_fscore / 20
        average_pairwise_score = average_pairwise_score / 20

        fmeasure.append(average_fscore)
        NMI.append(average_nmi)
        conslist.append(consnum)
        Homogeneity.append(average_hom)
        Completeness.append(average_comp)
        vmeasure.append(average_v)
        print("NMI result:", average_nmi)
        print("Fowlkes Mallows: ", average_fowlkes)
        print("Homogeneity, Completeness, vmeasure: ", average_hom, average_comp, average_v)
        print("F1: ", average_fscore)
        print("Pairwise F1: ", average_pairwise_score)

        Outfile.write("Number constraints: %s \n" % (consnum))
        Outfile.write("NMI: %s \n" % (average_nmi))
        Outfile.write("F measure: %s \n" % (average_fscore))
        Outfile.write("Pairwise F measure: %s \n" % (average_pairwise_score))
        Outfile.write("Fowlkes Mallows: %s \n" % (average_fowlkes))
        Outfile.write("Homogeneity, Completeness, vmeasure: %s \n" % (
            str(str(average_hom) + ", " + str(average_comp) + ", " + str(average_v))))
        consnum += 50
        Outfile.write("---------------------------------- \n \n")

    print("constraints num:",conslist)
    print("result NMI:",NMI)
    print("result F1 measure:",fmeasure)
    print("result Homogeneity:",Homogeneity)
    print("result Completeness:",Completeness)
    print("result vmeasure:",vmeasure)
    Outfile.write("Final Homogeneity: %s \n" % (Homogeneity))
    Outfile.write("Final Completeness: %s \n" % (Completeness))
    Outfile.write("Final vmeasure: %s \n" % (vmeasure))
    Outfile.write("Final NMI: %s \n" % (NMI))
    Outfile.write("Final F1 measure: %s \n" % (fmeasure))
    Outfile.close()

    #print("result fmeasure:",fmeasure)
    # db2= DBScanner(datavector,dataframe)
    # db2.dbscan(ml2, cl2)
    # db2.printcluster()
    # db2.getlabel()
    # f_measure2=db2.f_measure(dataframe)
    # db2.evaluation(dataframe)
    # db2.average_test(dataframe)
    # print("y2:", f_measure2)

    #datetime = time.strftime("%Y%m%d-%H%M%S")
    # plt.plot(conslist, fmeasure, color='g',marker="o")
    # plt.plot(conslist, NMI, color='orange',marker="x")
    # plt.xlabel('Constraint num')
    # plt.ylabel('Evaluation')
    # plt.title('Result of Evaluation with increasing Constraints')
    # plt.savefig( datetime+"Evaluation.png")
    # plt.show()

    plt.plot(conslist, NMI, color='g', marker="o", label="NMI")
    plt.plot(conslist, fmeasure, color='orange', marker="x", label="F-measure")
    # plt.plot(conslist, vmeasure2, color='blue',marker="v",label="V-measure")
    plt.legend(loc='center right')
    plt.xlabel('Constraint num')
    plt.ylabel('Evaluation')
    plt.savefig("Evaluation.png", dpi=300)
    plt.show()
if __name__ == '__main__':
    main()