#-------------------------------------------------------------------------------
# Name:        module3
# Purpose:
#
# Author:      lenovo
#
# Created:     24-07-2017
# Copyright:   (c) lenovo 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import re, powerlaw
import numpy, math, pylab
from numpy import *
from scipy.interpolate import spline
from nltk.corpus import stopwords, words
from nltk import pos_tag
import sys
from pylab import *
from itertools import chain, combinations
from collections import defaultdict, OrderedDict
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib, itertools
import preprocessor
import networkx as nx

def encode(text):
    """
    For printing unicode characters to the console.
    """
    return text.encode('utf-8')

def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

def tf(word, blob):
    return float(blob.count(word) / float(len(blob)))

def n_containing(word, blob, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, blob, bloblist):
    return math.log(len(bloblist) / float(1 + n_containing(word, blob, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, blob, bloblist)
def main():
    pass

if __name__ == '__main__':
    main()
usertweet=dict()
dusertweet=dict()
i=0
with open("CIKM3.txt", "r") as f:
    usertweet=eval(f.read())

##print usertweet


#Preprocessing of twitter feeds
j=1
for k,v in usertweet.iteritems():
    try:
        tempstorage=encode(usertweet[k])
    except:
        tempstorage=usertweet[k]
    preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.SMILEY, preprocessor.OPT.RESERVED)
    tempstorage=preprocessor.clean(tempstorage)
    tempstorage = re.sub(r"http\S+", "", tempstorage)
    tempstorage=re.sub(r'[^\x00-\x7F]+',' ', tempstorage)
    tempstorage = re.sub('[!$%@#^&*()~/.,:;\"\|\_\-\'\/]', '', tempstorage)
    tempstorage = ' '.join([word for word in tempstorage.split() if word not in stopwords.words("english")])
    tempstorage = ' '.join([word for word in tempstorage.split() if len(word)>2])
    temp=tempstorage.lower()
    dusertweet[k]=temp


##print dusertweet
countword=0
#Vocabulary of twitter feeds
vocabcorpus=[]
for k,v in dusertweet.iteritems():
    tempstore=dusertweet[k]
    listtemp=tempstore.split()
    for each in listtemp:
        countword=countword+1
        if each not in vocabcorpus:
            vocabcorpus.append(each)

##print vocabcorpus

#Creating dictionary of words
dictofwords=dict()
flag=1
for each in vocabcorpus:
            dictofwords[each]=flag
            flag=flag+1

##print dictofwords

#Indexing words in twitter feeds
templist=[]
duservaluekeys=dict()
j=1
for item in dusertweet.itervalues():
    s=item.split()
    for eachvalue in s:
        templist.append(dictofwords[eachvalue])
    duservaluekeys[j]=templist
    templist=[]
    j=j+1
##print duservaluekeys
flag=1
indexwisedictofwords=dict()
for k,v in dictofwords.iteritems():
    indexwisedictofwords[v]=k

#Listofedges
edgelist=[]
for eachvalue in duservaluekeys.itervalues():
    for i in range(1,len(eachvalue)):
        for j in range(i+1,len(eachvalue)):
            edge=(eachvalue[i],eachvalue[j])
            edgelist.append(edge)

##print edgelist

indexwisedictofwords=dict()
for k,v in dictofwords.iteritems():
    indexwisedictofwords[v]=k

#Weight of edges
weightdict=dict()
for each in edgelist:
    if each not in weightdict.keys():
        weightdict[each]=1
    else:
        weightdict[each]=weightdict[each]+1

##print weightdict



##lstwt=list(numpy.unique(weightdict.values()))
##print lstwt
##
##distdict=dict([(key, 0) for key in lstwt])
##
##for each,value in weightdict.iteritems():
##    if distdict[value]==0:
##        distdict[weightdict[each]]=1
##    else:
##        distdict[weightdict[each]]=distdict[weightdict[each]]+1
##
##newdictofvalues=dict()
##
##for x in range(1, lstwt[-1]):
##    if x in distdict.keys():
##        newdictofvalues[x]=distdict[x]
##    else:
##        newdictofvalues[x]=0
##
##lists=sorted(newdictofvalues.items())
##x, y = zip(*lists)
##newplot=plt.plot(x,y)
##plt.xlim(0,lstwt[-1])
##plt.ylim(0,max(newdictofvalues.values()))
##plt.xlabel('Edge Weight')
##plt.ylabel('Co-occurrence Frequency')
##
##plt.show(newplot)


##lstwt=list(weightdict.values())
##print lstwt
##
##fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
##fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
##fit.plot_pdf( color= 'b')
##
##print('alpha= ',fit.power_law.alpha,'  sigma= ',fit.power_law.sigma)
##
##
##
##
weightededgelist=[]
for (first,second) in weightdict.iterkeys():
    weightededgelist.append((first,second,weightdict[(first,second)]))

#print weightededgelist

G=nx.DiGraph()
for (u,v,d) in weightededgelist:
    G.add_edge(u,v,weight=d)

sumdeg=0
degdict=dict()
for each in G.nodes():
    deg=G.degree(each)
    degdict[each]=deg
    sumdeg=sumdeg+deg
##
##addk=(float)((float)(sumdeg)/(float)(len(G.nodes())))
##print "<k> UW DT"
##print addk
##print "***************************************************************************"

Z=G.to_undirected()
sumdeg=0
degdict=dict()
for each in Z.nodes():
    deg=Z.degree(each)
    degdict[each]=deg
    sumdeg=sumdeg+deg

addk=(float)((float)(sumdeg)/(float)(len(Z.nodes())))
print "<k> UW UD"
print addk
print "***************************************************************************"
##
##N=len(G.nodes())
##p=(float)((float)(len(G.edges()))/((N*(N-1))/2))
##ERG=nx.erdos_renyi_graph(N,p)
##sumdeg=0
##degdict=dict()
##for each in ERG.nodes():
##    deg=ERG.degree(each)
##    degdict[each]=deg
##    sumdeg=sumdeg+deg
##
##addk=(float)((float)(sumdeg)/(float)(len(ERG.nodes())))
##print "<kr> "
##print addk
##print "***************************************************************************"

wtdegdict=dict()
sumdeg=0
for each in G.nodes():
    deg=G.degree(each,weight="weight")
    wtdegdict[each]=deg
    sumdeg=sumdeg+deg
##
##addk=(float)((float)(sumdeg)/(float)(len(G.nodes())))
##print "<k> WT DT"
##print addk
##print "***************************************************************************"

wtdegdict=dict()
sumdeg=0
for each in Z.nodes():
    deg=Z.degree(each,weight="weight")
    wtdegdict[each]=deg
    sumdeg=sumdeg+deg

addk=(float)((float)(sumdeg)/(float)(len(Z.nodes())))
print "<k> WT UD"
print addk
print "***************************************************************************"



degdist=dict()
allval=list(sorted(set(degdict.values()),reverse=True))
degdist=dict([(key, 0) for key in allval])
for each,value in degdict.iteritems():
        degdist[value]=degdist[value]+1

lstwt=list(degdist.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "gamma UW UD"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"


degdist=dict()
allval=list(sorted(set(wtdegdict.values()),reverse=True))
degdist=dict([(key, 0) for key in allval])
for each,value in wtdegdict.iteritems():
        degdist[value]=degdist[value]+1

lstwt=list(degdist.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "gamma WT UD"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"

##avgneigh=dict()
##Z=G.to_undirected
##avgneigh=nx.k_nearest_neighbors(G, source='in')
##lstwt=list(avgneigh.values())
##fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
##fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
##fit.plot_pdf( color= 'b')
##print "Meu UW DT"
##print('alpha= ',fit.power_law.alpha)
##
##print "***************************************************************************"

##
##avgneigh=dict()
##Z=G.to_undirected()
##avgneigh=nx.average_neighbor_degree(Z)
##lstwt=list(avgneigh.values())
##fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
##fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
##fit.plot_pdf( color= 'b')
##print "Meu UW UD"
##print('alpha= ',fit.power_law.alpha)
##
##print "***************************************************************************"
##
####avgneigh=dict()
####Z=G.to_undirected
####avgneigh=nx.average_neighbor_degree(G, source='in', weight='weight')
####lstwt=list(avgneigh.values())
####fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
####fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
####fit.plot_pdf( color= 'b')
####print "Meu WT DT"
####print('alpha= ',fit.power_law.alpha)
####
####print "***************************************************************************"
##
##avgneigh=dict()
##Z=G.to_undirected()
##avgneigh=nx.average_neighbor_degree(Z, weight='weight')
##lstwt=list(avgneigh.values())
##fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
##fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
##fit.plot_pdf( color= 'b')
##print "Meu WT UD"
##print('alpha= ',fit.power_law.alpha)
##
##print "***************************************************************************"
##

addA=0
addB=0
addC=0
for (u,v) in Z.edges():
    x=Z.degree(u)
    y=Z.degree(v)
    addA=addA+(x+y)
    addB=addB+((float)((x+y)/2))
    addC=addC+((float)(((x*x)+(y*y))/2))

E=len(G.edges())
coeff=(float)((((float)(addA/E))-((float)((addB*addB)/(E*E))))/((float)(addC/E)-((float)((addB*addB)/(E*E)))))
print 'Tau UW'
print coeff

print "***************************************************************************"

addA=0
addB=0
addC=0
for (u,v) in Z.edges():
    x=Z.degree(u, weight='weight')
    y=Z.degree(v, weight='weight')
    addA=addA+(x+y)
    addB=addB+((float)((x+y)/2))
    addC=addC+((float)(((x*x)+(y*y))/2))

E=len(G.edges())
coeff=(float)((((float)(addA/E))-((float)((addB*addB)/(E*E))))/((float)(addC/E)-((float)((addB*addB)/(E*E)))))
print 'Tau WT'
print coeff

print "***************************************************************************"


##
##lstofdeg=sorted(list(set(wtdegdict.values())))
##nodesbydeg=dict()
##for each in lstofdeg:
##    templst=[]
##    for k,v in wtdegdict.iteritems():
##        if v==each:
##            templst.append(k)
##    nodesbydeg[each]=templst
##    templst=[]
##
##clusteringcoeff=dict()
##for k,v in nodesbydeg.iteritems():
##    clusteringcoeff[k]=nx.average_clustering(Z,v, weight='weight')
##lstwt=list(clusteringcoeff.values())
##fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
##fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
##fit.plot_pdf( color= 'b')
##print "Beta WT"
##print('alpha= ',fit.power_law.alpha)
##
##print "***************************************************************************"
##
##lst=[]
##weightknn=dict()
##for each in Z.nodes():
##    lstofneigh=Z.neighbors(each)
##    degofneighnode=0
##    count=0
##    for eachnode in lstofneigh:
##         degofneighnode=degofneighnode+Z.degree(eachnode)
##         count=count+1
##    calknn=(float)((float)(degofneighnode)/(float)(count))
##    weightknn[each]=calknn
##    lstofneigh=[]
##knndict=dict()
##lstofdeg=list(degdict.values())
##for each in lstofdeg:
##    count=0
##    addnode=0
##    for k,v in weightknn.iteritems():
##        if degdict[k]==each:
##            addnode=addnode+v
##            count=count+1
##    calknn=(float)((float)(addnode)/(float)(count))
##    knndict[each]=calknn
lstofdeg=sorted(list(set(degdict.values())))
nodesbydeg=dict()
for each in lstofdeg:
    templst=[]
    for k,v in degdict.iteritems():
        if v==each:
            templst.append(k)
    nodesbydeg[each]=templst
    templst=[]

clusteringcoeff=dict()
for k,v in nodesbydeg.iteritems():
    clusteringcoeff[k]=nx.average_clustering(Z,v)

#Hierarchichal Organization
lstwt=list(clusteringcoeff.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "Beta UW"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"

lstofdeg=sorted(list(set(wtdegdict.values())))
nodesbydeg=dict()
for each in lstofdeg:
    templst=[]
    for k,v in wtdegdict.iteritems():
        if v==each:
            templst.append(k)
    nodesbydeg[each]=templst
    templst=[]

clusteringcoeff=dict()
for k,v in nodesbydeg.iteritems():
    clusteringcoeff[k]=nx.average_clustering(Z,v, weight='weight')
lstwt=list(clusteringcoeff.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "Beta WT"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"

Z=G.to_undirected()
addk=nx.average_clustering(Z)
print "C UW"
print addk
print "***************************************************************************"

Z=G.to_undirected()
addk=nx.average_clustering(Z,weight='weight')
print "C WT"
print addk
print "***************************************************************************"


print "knn UW"
knn=nx.k_nearest_neighbors(G)
abc=(float)((float)(sum(knn.values()))/(float)(len(knn.values())))
print abc
lstwt=list(knn.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "Meu UW UD"
print('alpha= ',fit.power_law.alpha)

knn=dict()
print "***************************************************************************"
print "knn WT"
knn=nx.k_nearest_neighbors(G, weight='weight')
abc=(float)((float)(sum(knn.values()))/(float)(len(knn.values())))
print abc
lstwt=list(knn.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "Meu UW UD"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"

count=0
sum=0
for each in G.nodes():
    if G.out_degree(each)==0:
        denom=1
    else:
        denom=G.out_degree(each)
    sum=sum+(float)((float)(G.in_degree(each))/denom)
    count=count+1

IbyO=(float)(sum/count)
print "I/O UW"
print IbyO

print "****************************************************************************"


count=0
sum=0
for each in G.nodes():
    if G.out_degree(each, weight='weight')==0:
        denom=1
    else:
        denom=G.out_degree(each, weight='weight')
    sum=sum+(float)((float)(G.in_degree(each, weight='weight'))/denom)
    count=count+1

IbyO=(float)(sum/count)
print "I/O WT"
print IbyO