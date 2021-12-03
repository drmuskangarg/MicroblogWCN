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
    tempstorage = re.sub('[@#]', '', tempstorage)
    tempstorage = ' '.join([word for word in tempstorage.split() if word not in stopwords.words("english")])
    tempstorage = ' '.join([word for word in tempstorage.split() if len(word)>2])
    temp=tempstorage.lower()
    if j<1000:
        dusertweet[j]=temp
    j=j+1

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


N=len(G.nodes())
p=(float)((float)(len(G.edges()))/((N*(N-1))/2))
ERG=nx.erdos_renyi_graph(N,p, directed='True')
G=ERG
Z=G.to_undirected()
le=nx.adjacency_spectrum(Z)
addle=dict()
for each in list(le):
    if each in addle.keys():
        addle[each]=addle[each]+1
    else:
        addle[each]=1
plt.plot(223)
lists=sorted(addle.items(),reverse=True)
x, y = zip(*lists)
newpl=plt.plot(x,y,'ro', markersize=1)
plt.xscale('symlog')
plt.yscale('linear')
plt.xlabel('lambda')
plt.ylabel('Spectral Density')
plt.show(newpl)

sumdeg=0
degdict=dict()
for each in G.nodes():
    deg=G.degree(each)
    degdict[each]=deg
    sumdeg=sumdeg+deg

wtdegdict=dict()
sumdeg=0
for each in G.nodes():
    deg=G.degree(each,weight="weight")
    wtdegdict[each]=deg
    sumdeg=sumdeg+deg

inout=dict()
outin=dict()
for each,value in wtdegdict.iteritems():
    inout[G.in_degree(each,weight='weight')]=G.out_degree(each,weight='weight')
    outin[G.out_degree(each,weight='weight')]=G.in_degree(each,weight='weight')

lists=sorted(inout.items())
x, y = zip(*lists)
plt.subplot(221)
newplot=plt.plot(inout.keys(),inout.values(),'ro', markersize=1)
plt.xlim(0,max(inout.keys()))
plt.ylim(0,max(inout.values()))
plt.xlabel('In-degree')
plt.ylabel('Out-degree')

plt.show(newplot)

lists=sorted(outin.items())
x, y = zip(*lists)
plt.subplot(221)
newplot=plt.plot(x,y,'ro', markersize=1)
plt.xlim(0,max(outin.keys()))
plt.ylim(0,max(outin.values()))
plt.xlabel('Out-degree')
plt.ylabel('In-degree')

plt.show(newplot)



le=sorted(le,reverse=True)
addle=dict()
i=1
for each in list(le):
    addle[i]=each
    i=i+1

plt.subplot(223)
lists=sorted(addle.items(),reverse=False)
x, y = zip(*lists)
newplot=plt.plot(x,y,'ro', markersize=1)
plt.yscale('symlog')
##plt.xlim(0,len(le))
##plt.ylim(0,max(le))
plt.xlabel('Rank i')
plt.ylabel('Eigenvalues')

plt.show(newplot)

print max(le)

lst=[]
weightknn=dict()
for each in Z.nodes():
    lstofneigh=Z.neighbors(each)
    degofneighnode=0
    count=0
    for eachnode in lstofneigh:
         degofneighnode=degofneighnode+Z.degree(eachnode)
         count=count+1
    if count>=1:
        calknn=(float)((float)(degofneighnode)/(float)(count))
    weightknn[each]=calknn
    lstofneigh=[]
knndict=dict()
lstofdeg=list(degdict.values())
for each in lstofdeg:
    count=0
    addnode=0
    for k,v in weightknn.iteritems():
        if degdict[k]==each:
            addnode=addnode+v
            count=count+1
    calknn=(float)((float)(addnode)/(float)(count))
    knndict[each]=calknn
plt.subplot(222)
lists=sorted(knndict.items())
x, y = zip(*lists)
newplot=plt.plot(x,y,'ro',markersize=1)
#plt.xlim(0,max(knndict.keys()))
plt.xscale('log')
plt.yscale('log')
#plt.ylim(0,max(knndict.values()))
plt.xlabel('Degree k')
plt.ylabel('K-Nearest Neighbour')

plt.show(newplot)

degdist=dict()
allval=list(sorted(set(wtdegdict.values()),reverse=True))
degdist=dict([(key, 0) for key in allval])
for each,value in wtdegdict.iteritems():
        degdist[value]=degdist[value]+1
#Degree Distribution

plt.subplot(222)
lists=sorted(degdist.items(),reverse=True)
x, y = zip(*lists)
newplot=plt.plot(x,y,'ro', markersize=1)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Degree k')
plt.ylabel('Degree distribution')

plt.show(newplot)

#Edge Weight Distribution
wtdist=dict()
allval=list(sorted(set(weightdict.values()),reverse=True))
wtdist=dict([(key, 0) for key in allval])
for each,value in weightdict.iteritems():
        wtdist[value]=wtdist[value]+1
plt.subplot(222)
lists=sorted(wtdist.items(),reverse=True)
x, y = zip(*lists)
newplot=plt.plot(x,y,'ro', markersize=1)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Edge weight w')
plt.ylabel('Edge weight distribution')

plt.show(newplot)

#Clustering COefficient
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
plt.subplot(222)
lists=sorted(clusteringcoeff.items(),reverse=True)
x, y = zip(*lists)
newplot=plt.plot(x,y,'ro', markersize=1)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Degree k')
plt.ylabel('Clustering Coefficient')
plt.show(newplot)
print "******************************************************************************"

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
    clusteringcoeff[k]=nx.average_clustering(Z,v, weight='weight')
plt.subplot(222)
lists=sorted(clusteringcoeff.items(),reverse=True)
x, y = zip(*lists)
newplot=plt.plot(x,y,'ro', markersize=1)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Degree k')
plt.ylabel('Clustering Coefficient')
plt.show(newplot)
print "******************************************************************************"

