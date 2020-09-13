import csv
import math
import operator
import matplotlib.pyplot as plt
import random
from numpy import *
import numpy as np

# 感觉线性回归最基本的模型比不困难，关键是对w和b求偏导，之后通过梯度下降法使w和b接近最小误差，
# 卡了我很长时间的一个问题使每次更新不是将w和b减去求出后的w和b，而是减去他们两个之间的差。
class Linear_Model:#线性回归的自己手动实现
    featureList=[]
    labelList=[]
    step=0.001
    times=2000
    w=1
    b=1
    sumxy=0
    sumx=0
    sumy=0
    sumx2=0
    lenth=0
    # length=0
    lasterror=-1
    def __init__(self,Feature,Label,Step=0.05,times=200):
        self.featureList=Feature
        self.labelList=Label
        self.step=Step
        self.times=times
        self.length=len(self.featureList)
        print(self.featureList)
        print(self.labelList)
        for index in range(self.length):
            self.sumxy=self.sumxy+self.featureList[index]*self.labelList[index]
            self.sumx2=self.sumx2+self.featureList[index]*self.featureList[index]
            self.sumx=self.sumx+self.featureList[index]
            self.sumy=self.sumy+self.labelList[index]
    def calerror(self):
        error = 0
        for lenth in range(len(self.featureList)):
            error = error + (self.featureList[lenth] - self.w * self.labelList[lenth] - self.b) ** 2
        error=error/len(self.featureList)
        return error
    def fit(self):
        for sum in range(self.times):
            # self.lasterror=self.calerror()
            errorw=(self.sumxy-self.sumx*self.sumy/self.length)/(self.sumx2-self.sumx**2/self.length)
            errorb=(self.sumy-errorw*self.sumx)/self.length
            self.w=self.w-(self.w-errorw)*self.step
            self.b=self.b-(self.b-errorb)*self.step
            # print("{} {} {}".format(self.w,self.b,self.calerror()))
    def predict(self,d):
        return d*self.w+self.b


# 决策树的基本模型，对于决策树，感觉思路是比较清晰的，但是实现起来却不知道怎么做，之后参考了下别人的思路，然后写的
class   Decision_Tree:
    labellist=[]
    def __init__(self,dataset,caption):
        self.dataset=dataset
        # self.labellist=labellist
        self.caption=caption
        # self.entropy=self.calent(labellist)

    def calclass(self,labellist):
        labelcnt={}
        for vote in labellist:
            if vote not in labelcnt.keys():
                labelcnt[vote]=0
            labelcnt[vote]=labelcnt[vote]+1
        sortelabel=sorted(labelcnt.items(),key=operator.itemgetter(1),reverse=True)
        return sortelabel[0][0]
    def splitdataset(self,dataset,axis,value):
        sublist=[]
        for vec in dataset:
            if vec[axis]==value:
                reducevec=vec[:axis]
                reducevec.extend(vec[axis+1:])
                sublist.append(reducevec)
        return sublist


    def calent(self ,dataset):
        length=len(dataset)
        data={}
        entropy=0
        for obj in dataset:
            labellist=obj[-1]
            if labellist not in data.keys():
                data[labellist]=0
            data[labellist]=data[labellist]+1
        for index in data:
            prob=float(data[index])/length
            entropy=entropy-prob*math.log(prob,2)
        return entropy
    def fit(self,dataset,caption):
        entropy=[]

        labellist=[example[-1] for example in dataset]
        if labellist.count(labellist[0])==len(labellist):#这次递归这剩下一个标签
            return labellist[0]
        if len(dataset[0])==1:#这次只剩下一个种类
            return self.calclass(labellist)#这个种类中标签最多的那个就是
        featurecnt=len(dataset[0])-1
        rootentropy=self.calent(dataset)
        bestgain=0
        bestfeature=-1
        for cnt in range(featurecnt):
            singlelist=[example[cnt] for example in dataset]#取出第cnt列
            # for vec in featurelist[cnt]:
            #     singlelist.append(vec)
            uniqlist=set(singlelist)
            newentropy=0
            for value in uniqlist:
                subdataset=self.splitdataset(dataset,cnt,value)
                prob=len(subdataset)/float(len(dataset))
                newentropy=newentropy+prob*self.calent(subdataset)
            infogain=rootentropy-newentropy
            if infogain>bestgain:
                bestgain=infogain
                bestfeature=cnt
        bestfeaturelabel=caption[bestfeature]
        mytree={bestfeaturelabel:{}}
        del caption[bestfeature]
        featurevalue=[example[bestfeature] for example in dataset]
        uniqval=set(featurevalue)
        for value in uniqval:
            sublabel=caption[:]
            mytree[bestfeaturelabel][value]=self.fit(self.splitdataset(dataset,bestfeature,value),sublabel)
        return mytree




#k均值聚类的手动实现，比较简单，但是注意算距离的时候得**1/2，不然的话聚类中心重合
class K_Means:
    def distence(self,p1,p2):
        # print(p1,p2)
        return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    def means(self,arr):
        return np.array([np.mean([e[0] for e in arr]),np.mean([e[1] for e in arr])])
    def fastest(self,k_arr,arr):
        f=[0,0]
        max_d=0
        for e in arr:
            d=0
            for length in range(len(k_arr)):
                d=d+np.sqrt(self.distence(k_arr[length],e))
            if d>max_d:
                max_d=d
                f=e
        return f
    def fit(self,dataset):
        m=7
        k=np.random.randint(len(dataset)-1)
        clu_arr=[[]]
        k_arr=np.array([dataset[k]])
        for cnt in range(m-1):
            k=self.fastest(k_arr,dataset)
            k_arr=np.concatenate([k_arr,np.array([k])])
            clu_arr.append([])

        n=20
        cla_temp=clu_arr
        for cnt in range(n):
            for e in dataset:
                ki=0
                min_d=self.distence(e,k_arr[ki])
                for cnt2 in range(1,len(k_arr)):
                    if self.distence(e,k_arr[cnt2])<min_d:
                        min_d=self.distence(e,k_arr[cnt2])
                        ki=cnt2
                cla_temp[ki].append(e)
            for k in range(len(k_arr)):
                if n-1==cnt:
                    break
                k_arr[k]=self.means(cla_temp[k])
                cla_temp[k]=[]

        col =['HotPink','Aqua','Chartreuse','yellow','LightSalmon','red','black']
        for cnt in range(m):
            plt.scatter(k_arr[cnt][0],k_arr[cnt][1],linewidths=10,color=col[cnt])
            plt.scatter([e[0]for e in cla_temp[cnt]],[e[1]for e in cla_temp[cnt]],color=col[cnt])
        plt.show()

#支持向量机部分，实在看不懂，虽然数学推导大致的意思知道，但是感觉这python实现数学公式好像基本没用到，等老师讲过后补下好了
#暂时通过sklearn熟悉下svm好了
    # from numpy import *
    # import numpy as np
    # import csv
    # import matplotlib.pyplot as plt
    #
    # def load_set_data(file_name):
    #     data_mat = []
    #     label_mat = []
    #     n = 0
    #
    #     fr = open(file_name)
    #
    #     for line in fr.readlines():
    #         line_arr = line.strip().split(",")
    #         data_mat.append([float(line_arr[0]), float(line_arr[1])])
    #         label_mat.append(float(line_arr[2]))
    #         n += 1
    #
    #     return data_mat, label_mat, n
    #
    # def select_j_rand(i, m):
    #     j = i
    #     while (j == i):
    #         j = int(random.uniform(0, m))
    #
    #     return j
    #
    # def clip_alpha(aj, H, L):
    #     if aj > H:
    #         aj = H
    #     if aj < L:
    #         aj = L
    #
    #     return aj
    #
    # def smo_simple(data_matin, class_labele, C, toler, max_iter):
    #     data_matrix = mat(data_matin)  # 将输入列表转成矩阵
    #     label_mat = mat(class_labele).transpose()  # 将训练数据转成列向量
    #     b = 0
    #     m, n = shape(data_matrix)
    #     alphas = mat(zeros((m, 1)))
    #     iter = 0
    #     while (iter < max_iter):
    #         alpha_pirs_change = 0
    #         for i in range(m):
    #             fxi = float(multiply(alphas, label_mat).T * np.matrix.dot((data_matrix), (data_matrix[i, :].T))) + b
    #             ei = fxi - float(label_mat[i])
    #             if not (alphas[i] >= 0 and label_mat[i] * fxi - 1 >= 0 and alphas[i] * (
    #                     label_mat[i] * fxi - 1) == 0):  # 不满足KKT
    #                 j = select_j_rand(i, m)  # 随机选择aj且i != j，相当于随机选择ai和aj
    #                 fxj = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
    #                 ej = fxj - float(label_mat[j])
    #                 alpha_iold = alphas[i].copy()  # 保存alphai更新前的值
    #                 alpha_jold = alphas[j].copy()  # 保存alphaj更新前的值
    #                 # 求解alphaj的上下边界
    #                 if (label_mat[i] != label_mat[j]):
    #                     L = max(0, alpha_jold - alpha_iold)
    #                     H = min(C, C + alpha_jold + alpha_iold)
    #                 else:
    #                     L = max(0, alpha_jold + alpha_iold - C)
    #                     H = min(C, alpha_iold + alpha_jold)
    #                 if (L == H):
    #                     # print("L == H")
    #                     continue
    #                 eta = 2 * data_matrix[i, :] * data_matrix[j, :].T \
    #                       - data_matrix[i, :] * data_matrix[i, :].T - data_matrix[j, :] * data_matrix[j, :].T
    #                 if (eta > 0):
    #                     # print("eta > 0")
    #                     continue
    #                 alphas[j] = alpha_jold - (label_mat[j] * (ei - ej) * 1.0 / eta)
    #                 alphas[j] = clip_alpha(alphas[j], H, L)
    #                 if (abs(alphas[j] - alpha_jold) < 0.00001):
    #                     # print("j not moving enough")
    #                     continue
    #                 alphas[i] = alpha_iold + (label_mat[i] * label_mat[j] * (alpha_jold - alphas[j]))
    #                 b1 = b - ei - label_mat[i] * (alphas[i] - alpha_iold) * (data_matrix[i, :] * data_matrix[i, :].T) \
    #                      - label_mat[j] * (alphas[j] - alpha_jold) * (data_matrix[i, :] * data_matrix[j, :].T)
    #                 b2 = b - ej - label_mat[i] * (alphas[i] - alpha_iold) * (data_matrix[i, :] * data_matrix[j, :].T) \
    #                      - label_mat[j] * (alphas[j] - alpha_jold) * (data_matrix[j, :] * data_matrix[j, :].T)
    #                 if (alphas[i] > 0 and alphas[i] < C):
    #                     b = b1
    #                 elif (alphas[j] > 0 and alphas[j] < C):
    #                     b = b2
    #                 else:
    #                     b = (b1 + b2) / 2.0
    #                 alpha_pirs_change += 1
    #                 # print("iter: %d i: %d, paris changed % d" % (iter, i, alpha_pirs_change))
    #         if (alpha_pirs_change == 0):
    #             iter += 1
    #         else:
    #             iter = 0
    #             # print("iteration number: %d" % iter)
    #     return b, alphas
    #
    # def show_experiment_plot(alphas, data_list_in, label_list_in, b, n):
    #     data_arr_in = array(data_list_in)
    #     label_arr_in = array(label_list_in)
    #     alphas_arr = alphas.getA()
    #     data_mat = mat(data_list_in)
    #     label_mat = mat(label_list_in).transpose()
    #
    #     i = 0
    #     weights = zeros((2, 1))
    #     while (i < n):
    #         if (label_arr_in[i] == -1):
    #             plt.plot(data_arr_in[i, 0], data_arr_in[i, 1], "ob")
    #         elif (label_arr_in[i] == 1):
    #             plt.plot(data_arr_in[i, 0], data_arr_in[i, 1], "or")
    #         if (alphas_arr[i] > 0):
    #             plt.plot(data_arr_in[i, 0], data_arr_in[i, 1], "oy")
    #             weights += multiply(alphas[i] * label_mat[i], data_mat[i, :].T)
    #         i += 1
    #
    #     x = arange(-2, 12, 0.1)
    #     y = []
    #     for k in x:
    #         y.append(float(-b - weights[0] * k) / weights[1])
    #
    #     plt.plot(x, y, '-g')
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.show()
    #
    # def main():
    #     data_list, label_list, n = load_set_data("svm.txt")
    #     b, alphas = smo_simple(data_list, label_list, 0.6, 0.001, 40)
    #     b_data = b
    #     show_experiment_plot(alphas, data_list, label_list, b_data, n)
    #
    # main()
# l= Linear_Model([1,1],[1,1])
#支持向量机的sklearn部分
# from sklearn import svm
#
# data_mat = []
# label_mat = []
# n = 0
#
# fr = open("svm.txt","r")
#
# for line in fr.readlines():
#     line_arr = line.strip().split(",")
#     data_mat.append([float(line_arr[0]), float(line_arr[1])])
#     label_mat.append(float(line_arr[2]))
#     n += 1
#
# model=svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovr')
# model.fit(data_mat,label_mat)
# print(model.predict([[1,1]]))

#BP神经网络的简单实现，感觉并不是特别难，先正向跑一边，然后根据书上的公式反向跑一边，书上的公式自己推过了，但是这个程序有个小问题，就是当循环的
#遍数较少时，可能会不能成功学习
class backpropagation:
    def getlink(self,sum1,sum2,fill=0.0):
        mat=[]
        for i in range(sum1):
            mat.append([fill]*sum2)
        return mat
    def rand(self,d1,d2):
        return (d2-d1)*random.random()+d1
    def sigmoid(self,data):
        return 1.0/(1.0+math.exp(-data))
    def segmoid_derivatiive(self,data):
        return data*(1-data)
    def __init__(self,inl,hnl,onl):
        inl=inl+1
        self.num_input=inl
        self.num_output=onl
        self.num_hide=hnl
        self.input_set=[1.0]*inl
        self.hide_set=[1.0]*hnl
        self.output_set=[1.0]*onl
        self.linkih=self.getlink(inl,hnl)
        self.linkho=self.getlink(hnl,onl)
        for i in range(inl):
            for h in range(hnl):
                self.linkih[i][h]=self.rand(-0.2,0.2)
        for h in range(hnl):
            for o in range(onl):
                self.linkho[h][o]=self.rand(-2.0,2.0)
        self.correctih=self.getlink(inl,hnl)
        self.correctho=self.getlink(hnl,onl)

    def predict(self,inputs):
        for i in range(self.num_input-1):
            self.input_set[i]=inputs[i]
        for h in range(self.num_hide):
            sum=0.0
            for i in range(self.num_input):
                sum=sum+self.input_set[i]*self.linkih[i][h]
            self.hide_set[h]=self.sigmoid(sum)
        for o in range(self.num_output):
            sum=0.0
            for h in  range(self.num_hide):
                sum=sum+self.hide_set[h]*self.linkho[h][o]
            self.output_set[o]=self.sigmoid(sum)
        return self.output_set[:]

    def bp(self,data,label,learn,correct):
        self.predict(data)
        output_deltas=[0.0]*self.num_output
        for o in range(self.num_output):
            error=label[o]-self.output_set[o]
            output_deltas[o]=self.segmoid_derivatiive(self.output_set[o])*error#g(o)
        hidden_deltas=[0.0]*self.num_hide
        for h in range(self.num_hide):
            error=0.0
            for o in range(self.num_output):
                error=error+output_deltas[o]*self.linkho[h][o]
            hidden_deltas[h]=error*self.segmoid_derivatiive(self.hide_set[h])#e(h)
        for h in range(self.num_hide):
            for o in range(self.num_output):
                change=learn*output_deltas[o]*self.hide_set[h]
                self.linkho[h][o]=self.linkho[h][o]+change+correct*self.correctho[h][o]
                self.correctho[h][o]=change
        for i in range(self.num_input):
            for h in range(self.num_hide):
                change=learn*hidden_deltas[h]*self.input_set[i]
                self.linkih[i][h]=self.linkih[i][h]+change+self.correctih[i][h]*correct
                self.correctih[i][h]=change
        error=0.0
        for o in range(self.num_output):
            error=error+0.5*(label[o]-self.output_set[o])**2
        return error
    def fit(self,data,label,limit=100000,learn=0.05,correct=0.1):
        for cnt in range(limit):
            error=0.0
            for value in range(len(data)):
                datavalue=data[value]
                labelvalue=label[value]
                error=error+self.bp(datavalue,labelvalue,learn,correct)

#朴素Baysian，通过数学的统计加上拉普拉斯修正获取，之后只需要比较两者的几率就能推测
class Baysian:
    pset={}
    def cal(self,data1, data2, value1, value2):
        dirc = {}
        sublist = []
        sum1 = 0
        sum2 = 0
        for value in range(len(data1)):
            if data2[value ] == value2:
                sum1 = sum1 + 1
                if data1[value] == value1:
                    sum2 = sum2 + 1
        return (sum2 + 1) / (sum1 + len(set(data1)))
    def fit(self,featurelist,labellist):
        labelset = {}
        for cnt in range(1, len(labellist)):
            if labellist[cnt] not in labelset:
                labelset[labellist[cnt]] = 0
            labelset[labellist[cnt]] = labelset[labellist[cnt]] + 1


        for cnt in range(0, len(featurelist[0])):
            sublist = []
            subdict = {}
            bj = []
            for value in range(1, len(featurelist)):
                bj.append(featurelist[value][cnt])

            bjset = set(bj)

            for value in bjset:
                set1 = {}
                for label in labelset:
                    set1[label] = self.cal(bj, labellist, value, label)
                self.pset[value] = set1

    def predict(self,feature,labellist):
        p={}
        for label in labellist:
            per=1
            for value in feature:
                per=per*self.pset[value][label]
            p[label]=per
        bestkey=0
        bestpercent=0
        for keys in p:
            if p[keys]>bestpercent:
                bestkey=keys
                bestpercent=p[keys]
        return bestkey




















