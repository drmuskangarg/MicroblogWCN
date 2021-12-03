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
with open("CIKM3000.txt", "r") as f:
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


Z=G.to_undirected(G)
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

print "Length: "
print countword
print "***************************************************************************"

print "N: "
print len(G.nodes())
print "***************************************************************************"

print "E: "
print len(weightdict.keys())
print "***************************************************************************"

sumdeg=0
degdict=dict()
for each in G.nodes():
    deg=G.degree(each)
    degdict[each]=deg
    sumdeg=sumdeg+deg

addk=(float)((float)(sumdeg)/(float)(len(G.nodes())))
print "<k> UW DT"
print addk
print "***************************************************************************"

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

addk=(float)((float)(sumdeg)/(float)(len(G.nodes())))
print "<k> WT DT"
print addk
print "***************************************************************************"

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

Gc = max(nx.weakly_connected_component_subgraphs(G), key=len)
addk=nx.average_shortest_path_length(Gc)
print "L DT"
print addk
print "***************************************************************************"

Z=G.to_undirected()
Gc = max(nx.connected_component_subgraphs(Z), key=len)
addk=nx.average_shortest_path_length(Gc)
print "L UD"
print addk
print "***************************************************************************"

N=len(G.nodes())
p=(float)((float)(len(G.edges()))/((N*(N-1))/2))
ERG=nx.erdos_renyi_graph(N,p)

Gc = max(nx.connected_component_subgraphs(ERG), key=len)
addk=nx.average_shortest_path_length(Gc)
print "Lr UD"
print addk
print "***************************************************************************"

N=len(G.nodes())
p=(float)((float)(len(G.edges()))/((N*(N-1))/2))
ERG=nx.erdos_renyi_graph(N,p, directed=True)

Gc = max(nx.weakly_connected_component_subgraphs(ERG), key=len)
addk=nx.average_shortest_path_length(Gc)
print "Lr DT"
print addk
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

N=len(G.nodes())
p=(float)((float)(len(G.edges()))/((N*(N-1))/2))
ERG=nx.erdos_renyi_graph(N,p)
addk=nx.average_clustering(ERG)
print "Cr"
print addk
print "***************************************************************************"

N=len(G.nodes())
p=(float)((float)(len(G.edges()))/((N*(N-1))/2))
ERG=nx.erdos_renyi_graph(N,p)
addk=nx.average_clustering(ERG)
print "Cr"
print addk
print "***************************************************************************"


lstwt=list(weightdict.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "weighted coeff"
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

avgneigh=dict()
Z=G.to_undirected
avgneigh=nx.k_nearest_neighbors(G, source='in')
lstwt=list(avgneigh.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "Meu UW DT"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"


avgneigh=dict()
Z=G.to_undirected()
avgneigh=nx.average_neighbor_degree(Z)
lstwt=list(avgneigh.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "Meu UW UD"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"

avgneigh=dict()
Z=G.to_undirected
avgneigh=nx.average_neighbor_degree(G, source='in', weight='weight')
lstwt=list(avgneigh.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "Meu WT DT"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"

avgneigh=dict()
Z=G.to_undirected()
avgneigh=nx.average_neighbor_degree(Z, weight='weight')
lstwt=list(avgneigh.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "Meu WT UD"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"


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

le=nx.laplacian_spectrum(Z)

le=nx.laplacian_spectrum(Z)

sortedle=list(sorted(le,reverse=True))
print 'Lambda1',sortedle[0]
print 'lambda2',sortedle[1]
print 'lambda3',sortedle[2]
print 'lambda4',sortedle[3]
print 'last lambda',sortedle[-1]
print 'Total lambda',len(sortedle)
print "****************************************************************************"



inout=dict()
outin=dict()
for each,value in wtdegdict.iteritems():
    inout[G.in_degree(each,weight='weight')]=G.out_degree(each,weight='weight')
    outin[G.out_degree(each,weight='weight')]=G.in_degree(each,weight='weight')

lists=sorted(inout.items())
x, y = zip(*lists)
plt.subplot(225)
newplot=plt.plot(inout.keys(),inout.values(),'ro', markersize=1)
plt.xlim(0,max(inout.keys()))
plt.ylim(0,max(inout.values()))
plt.xlabel('indeg')
plt.ylabel('outdeg')

plt.show(newplot)

lists=sorted(outin.items())
x, y = zip(*lists)
plt.subplot(221)
newplot=plt.plot(x,y,'ro', markersize=1)
plt.xlim(0,max(outin.keys()))
plt.ylim(0,max(outin.values()))
plt.xlabel('outdeg')
plt.ylabel('indeg')

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
plt.xlabel('rank i')
plt.ylabel('eigenvalues')

plt.show(newplot)

print max(le)

##alldeg=sorted(list(degdict.keys()))
##plt.subplot(222)
##lists=sorted(wtdegdict.items(),reverse=True)
##x, y = zip(*lists)
##newpl=plt.plot(x,y)
##plt.yscale('log')
##plt.xlim(0,max(wtdegdict.keys()))
##plt.ylim(0,max(wtdegdict.values()))
##plt.xlabel('edgeweight i')
##plt.ylabel('frequency')
##
##plt.show(newpl)
##
##print max(le)
##degass=nx.degree_assortativity_coefficient(G, x='out',y='out')
##print degass
##
##print "******************************"
lst=[]
weightknn=dict()
for each in Z.nodes():
    lstofneigh=Z.neighbors(each)
    degofneighnode=0
    count=0
    for eachnode in lstofneigh:
         degofneighnode=degofneighnode+Z.degree(eachnode)
         count=count+1
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
plt.xlabel('Degree (K)')
plt.ylabel('K-nn')

plt.show(newplot)

#Degree Distribution

plt.subplot(222)
lists=sorted(degdist.items(),reverse=True)
x, y = zip(*lists)
newplot=plt.plot(x,y,'ro', markersize=1)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('degree k')
plt.ylabel('degree distribution')

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
plt.xlabel('degree k')
plt.ylabel('C(k) UW')
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
plt.xlabel('degree k')
plt.ylabel('C(k) WT')
plt.show(newplot)
print "******************************************************************************"




#Hierarchichal Organization
lstwt=list(clusteringcoeff.values())
fit = powerlaw.Fit(numpy.array(lstwt)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "Beta"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"


fit = powerlaw.Fit(numpy.array(le)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')
print "Beta"
print('alpha= ',fit.power_law.alpha)

print "***************************************************************************"

print "Average degree"
Ad=(float)(sum(degdict.values()) / (float)(len(degdict.values())))
print Ad
print "***************************************************************************"

print "Average weighted degree"
Awd=(float)(sum(wtdegdict.values()) / (float)(len(wtdegdict.values())))
print Awd
print "***************************************************************************"

print "Diameter"
print nx.diameter(G)


##xnew=linspace(min(knndict.keys()),max(knndict.keys()),500)
##powersmooth=spline(knndict.keys(),knndict.values(),xnew)
##plt.plot(xnew,powersmooth)
##plt.show()

##
##newG=nx.Graph()
##maxvalueofedge=max(weightdict.values())
##for u,v,d in weightededgelist:
##    wt=(float)((float)(d)/(float)(maxvalueofedge))
##    newG.add_edge(u,v,weight=wt)
##
##getedges=nx.minimum_spanning_edges(newG)
##lstedges=list(getedges)
##print sorted(lstedges)
##
##
##nx.draw_networkx(G, with_labels=True)
##plt.show(G)
##
##seqev=nx.eigenvector_centrality(G)
##print seqev

##x=(int)(math.ceil((float)(0.2*(float)(len(edgelist)))))
##x=(int)(math.ceil(math.sqrt(len(edgelist))))
##
##dictionary=dict()
##dictionary=OrderedDict(sorted(weightdict.items(), key=lambda x:x[1], reverse=True))
##best = (list(dictionary.keys())[:x])
##print best

##print x
##
##summ=0
##addfinaledgedict=dict()
##for each,value in dictionary.iteritems():
##    if summ<=x:
##            addfinaledgedict[each]=value
##    summ=summ+v
##
##print addfinaledgedict
##for k,v in duservaluekeys.iteritems():
##    for each in v:
##        print indexwisedictofwords[each]
##        print G.degree(each)
##    print "************************************************************"

###Find Degree of each node
##degreedict=dict()
##for each in D.nodes():
##    degreedict[each]=D.degree(each)
##
##print degreedict



