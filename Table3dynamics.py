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
    for i in range(1,len(eachvalue)-1):
            edge=(eachvalue[i],eachvalue[i+1])
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


addk=0
count=0
for g in nx.weakly_connected_component_subgraphs(G):
    addk=addk+(float)(nx.average_shortest_path_length(g))
    count=count+1
addk=(float)((float)(addk)/count)
print "L DT Full"
print addk
print "***************************************************************************"

Z=G.to_undirected()
addk=0
count=0
for g in nx.connected_component_subgraphs(Z):
    addk=addk+(float)(nx.average_shortest_path_length(g))
    count=count+1
addk=(float)((float)(addk)/count)
print "L UD Full"
print addk
print "***************************************************************************"

N=len(G.nodes())
p=(float)((float)(len(G.edges()))/((N*(N-1))/2))
ERG=nx.erdos_renyi_graph(N,p)

addk=0
count=0
for g in nx.connected_component_subgraphs(ERG):
    if len(g.nodes())>1:
        addk=addk+(float)(nx.average_shortest_path_length(g))
    count=count+1
addk=(float)((float)(addk)/count)
print "Lr UD Full"
print addk
print "***************************************************************************"

N=len(G.nodes())
p=(float)((float)(len(G.edges()))/((N*(N-1))/2))
ERG=nx.erdos_renyi_graph(N,p, directed=True)

addk=0
count=0
for g in nx.weakly_connected_component_subgraphs(ERG):
    if len(g.nodes())>1:
        addk=addk+(float)(nx.average_shortest_path_length(g))
    count=count+1
addk=(float)((float)(addk)/count)
print "Lr DT Full"
print addk
print "***************************************************************************"

Z=G.to_undirected()
Gc = max(nx.connected_component_subgraphs(Z), key=len)
addk=nx.average_shortest_path_length(Gc)
numofnodesUD=len(Gc.nodes())
print "L UD"
print addk
print "***************************************************************************"

Z=G.to_undirected()
Gc = max(nx.weakly_connected_component_subgraphs(G), key=len)
addk=nx.average_shortest_path_length(Gc)
numofnodesDT=len(Gc.nodes())
print "L DT"
print addk
print "***************************************************************************"

N=numofnodesUD
p=(float)((float)(len(G.edges()))/((N*(N-1))/2))
ERG=nx.erdos_renyi_graph(N,p)

Gc = max(nx.connected_component_subgraphs(ERG), key=len)
addk=nx.average_shortest_path_length(Gc)
print "Lr UD"
print addk
print "***************************************************************************"

N=numofnodesDT
p=(float)((float)(len(G.edges()))/((N*(N-1))/2))
ERG=nx.erdos_renyi_graph(N,p, directed=True)

Gc = max(nx.weakly_connected_component_subgraphs(ERG), key=len)
addk=nx.average_shortest_path_length(Gc)
print "Lr DT"
print addk
print "***************************************************************************"



