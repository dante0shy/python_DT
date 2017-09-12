import random
import numpy as np
from numpy import *
from collections import defaultdict
class DT(object):

    def __init__(self,attrset=set()):
        self.nextNodes = {}
        self.pre = -1
        self.attrset = attrset
        self.pickedattr = -1


    def getpre(self,x,y):
        counter = 0
        preDict = defaultdict(int)
        for idx,xi in enumerate(x):
            preDict[str(y[idx])] += 1
            counter +=1
        pre = int(preDict.keys()[0])
        max = preDict[preDict.keys()[0]]
        for idx in preDict.keys():
            if preDict[idx]>=max:
                pre = int(idx)
                max = preDict[idx]
        return pre

    def get_entropy(self,x,y):
        numdict = defaultdict(int)
        for yi in y:
            numdict[str(yi)] +=1
        entropy = 0.
        total = float(len(y))
        import math
        for num in numdict.values():
            entropy += - (float(num)/total)* math.log10( (float(num)/total))
        return entropy

    def addNode(self,x , y):
        if x[0].shape[0] == len(self.attrset):
            self.pre = self.getpre(x,y)
        pre = y[0]
        counter = 0
        for yi in y:
            if not yi ==pre:
                break
            else:
                counter +=1
        if counter == y.shape[0]:
            self.pre = pre
            return
        counter = 0
        x0 = x[0]
        preDict = defaultdict(int)
        for idx,xi in enumerate(x):
            if not (x0 ==xi).all():
                break
            else:
                preDict[str(y[idx])] += 1
                counter +=1
        if counter == x.shape[0]:
            pre = int(preDict.keys()[0])
            max = preDict[preDict.keys()[0]]
            for idx in preDict.keys():
                if preDict[idx]>=max:
                    pre = int(idx)
                    max = preDict[idx]
            self.pre = pre
            return

        entropy = self.get_entropy(x,y)
        avAttribute = [a for a in range(x[0].shape[0]) if a not in self.attrset]

        pickedAttr = 0
        gain = 0
        for attr in avAttribute:
            kinds = set(x[:,attr])
            gainAttr = entropy
            for kind in kinds:
                subx = []
                suby=[]
                for idx,xi in enumerate(x):
                    if xi[attr] == kind:
                        subx.append(xi)
                        suby.append(y[idx])
                if len(suby) == 0:
                    continue
                gainAttr -= (float(len(suby))/float(len(y)))*self.get_entropy(np.array(subx),np.array(suby))
            print 'level {} - attr {}: {}'.format(len(self.attrset),attr,gainAttr)
            if gainAttr > gain :
                pickedAttr = attr
                gain = gainAttr
        self.pickedattr = pickedAttr
        kinds = set(x[:,self.pickedattr])

        if len(self.attrset)>0:
            tmp = [a for a in self.attrset]
            tmp.append(pickedAttr)
            attrset = set(tmp)
        else:
            attrset = set([pickedAttr])
        for kind in kinds:
            subx = []
            suby=[]
            for idx,xi in enumerate(x):
                if xi[pickedAttr] == kind:
                    subx.append(xi)
                    suby.append(y[idx])
            if len(y)==0:
                continue
            self.nextNodes[str(kind)] = DT(attrset)
            self.nextNodes[str(kind)].addNode(np.array(subx),np.array(suby))
        return

    def fit(self,x,y):
        if not x.shape[0] == y.shape[0]:
            raise ValueError("error!")
        pre = y[0]
        counter = 0
        for yi in y:
            if not yi ==pre:
                break
            else:
                counter +=1
        if counter == y.shape[0]:
            self.pre = pre
            return

        x0 = x[0]
        counter = 0
        preDict = defaultdict(int)
        for idx,xi in enumerate(x):
            if not (x0 ==xi).all():
                break
            else:
                preDict[str(y[idx])] += 1
                counter +=1
        if counter == x.shape[0]:
            pre = int(preDict.keys()[0])
            max = preDict[preDict.keys()[0]]
            for idx in preDict.keys():
                if preDict[idx]>=max:
                    pre = int(idx)
                    max = preDict[idx]
            self.pre = pre
            return

        self.addNode(x,y)
        return

    def getpreict(self,x):
        result = 0
        if self.pickedattr ==-1:
            return self.pre
        else:
            a = str(x[self.pickedattr])
            result = self.nextNodes[str(x[self.pickedattr])].getpreict(x)
        return  result

    def preict(self,x):
        ypre = []
        for xi in x:
            ypre.append(self.getpreict(xi))

        return np.array(ypre)

    def gettree(self,level,father = 0):
        print 'level{}-{} father : {}'.format(level,self.pickedattr,father)
        for child in self.nextNodes.values():
            child.gettree(level+1,str(level)+'-'+str(self.pickedattr))
if __name__ == '__main__':
    x = np.array([
        [1,1,1,1,1,1],
        [2,1,1,1,1,1],
        [1,1,2,1,1,1],
        [3,1,1,1,1,1],
        [1,2,1,1,2,2],
        [2,2,1,2,2,2],
        [3,1,1,3,3,2],
        [1,2,1,2,1,1],
        [3,2,2,2,1,1],
        [2,2,1,1,2,2],
        [3,1,1,3,3,1],
        [2,2,1,1,2,1],
        [2,2,2,2,2,1],
        [1,3,3,1,3,2],
        [3,3,3,3,3,1]
    ])
    y = np.array([
        1,1,1,1,1,0,0,0,0,0,0,1,1,0,0])
    print np.array(y)
    c = DT()

    c.fit(x,y)
    print 'train finish'
    x = np.array([
        [1,1,2,2,2,1],
        [2,1,2,1,1,1]
    ])
    y = np.array([0,1])
    yPre = c.preict(x)
    print 'predict finish'
    print 'real:'
    print y
    print 'pre:'
    print yPre
    print c.gettree(0)

