# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class GA():
    def __init__(self, model, popsize, max_feat, iter_=50):
        self.model = model
        self.popsize = popsize
        self.max_feat = max_feat
        self.iter_ = iter_
        
    def popinit(self,X):
        nfeat = X.shape[1]
        listpop = []
        for i in range(self.popsize):
            pop_ = np.random.choice(nfeat, self.max_feat, replace=False)
            pop_ = list(pop_)
            pop_.sort()
            listpop.append(pop_)
        return listpop
        
    def fitness(self,pop,x,y):
        cv = KFold(n_splits = 10, random_state=None, shuffle=False)
        save_fit = []
        acc = []
        for i in range(len(pop)):
            x_slice = x.iloc[:,pop[i]]
            acs = cross_val_score(self.model, x_slice, y,scoring='accuracy', cv=cv, n_jobs=-1)
            acc.append(np.average(acs))
            mse = cross_val_score(self.model, x_slice, y,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
            mse = np.average(mse)
            mse = mse*-1
            fitness = 1/ (mse + 1e-10)
            save_fit.append(fitness)
        # print(acc)
        max_acc = max(acc)
        avg_acc = np.average(acc)
        # print(acc)
        return save_fit, max_acc, avg_acc
    
    def parent_select(self, pop, fit):
      sum_fit = np.sum(fit)
      prob = [x/sum_fit for x in fit]
      prob_range = []
      sum_ = 0
      for i in range(len(prob)):
        sum_ += prob[i]
        prob_range.append(sum_)
      # select first parent
      for i in range(len(prob)):
          rand = np.random.rand()
          if rand <= prob_range[i]:
            parent_1 = pop[i]
            break
      # select second parent
      parent_2 = parent_1
      while (parent_2 == parent_1):
        for i in range(len(prob)):
            rand = np.random.rand()
            if rand <= prob_range[i]:
                parent_2 = pop[i]
                break
      return parent_1, parent_2
    
    def cross_over(self, parent_1, parent_2):
        point = len(parent_1) - 1
        point_list = list(range(point))
        point_list = [x+1 for x in point_list]
        i=0
        while True:
            point_sel = np.random.choice(point_list, 1)[0]
            child_1 = parent_1[:point_sel] + parent_2[point_sel:]
            child_2 = parent_2[:point_sel] + parent_1[point_sel:]
            if(len(child_1) == len(set(child_1)) and len(child_2)==len(set(child_2)) and i==10):
                break
            elif(i>10):
                break
            i+=1
        return child_1, child_2
    
    def mutation(self, child_1, child_2, x):
        max_feat_list = list(range(x.shape[1]))
        feat_sel_child_1 = set(max_feat_list).difference(set(child_1))
        feat_sel_child_1 = list(feat_sel_child_1)
        feat_sel_child_2 = set(max_feat_list).difference(set(child_2))
        feat_sel_child_2 = list(feat_sel_child_2)
        mut_rate = 1/self.max_feat
        for i in range(len(child_1)):
            rand_1 = np.random.rand()
            rand_2 = np.random.rand()
            if rand_1 <= mut_rate:
                new_feat = np.random.choice(feat_sel_child_1, 1)[0]
                child_1[i] = new_feat
            if rand_2 <= mut_rate:
                new_feat = np.random.choice(feat_sel_child_2, 1)[0]
                child_2[i] = new_feat
        if(len(child_1) != len(set(child_1)) and len(child_2)!=len(set(child_2))):
            child_1, child_2 = self.mutation(child_1, child_2, x)
        return child_1, child_2
    
    def sort(self, pop, fit):
        tmp_dict = {'fitness': fit}
        df = pd.DataFrame(tmp_dict)
        df.reset_index(inplace=True)
        df.sort_values('fitness', ascending=False, inplace=True)
        idx_ = df.index.values.tolist()
        fit = df['fitness'].values.tolist()
        new_pop = []
        for i in idx_:
            new_pop.append(pop[i])
        return new_pop, fit
    
    def fit(self,x,y):
        mse_list = []
        avg_acc_list = []
        pop = self.popinit(x)
        fitness, max_acc, avg_acc = self.fitness(pop,x,y)
        print('max acc generation - 0 : ',max_acc,' for max feat = ',self.max_feat)
        avg_acc_list.append(avg_acc)
        mses = [1/x for x in fitness]
        mse_list.append(np.average(mses))
        
        for i in range(self.iter_-1):
            parent_1, parent_2 = self.parent_select(pop, fitness)
            child_1, child_2 = self.cross_over(parent_1, parent_2)
            child_1, child_2 = self.mutation(child_1, child_2, x)
            pop, fitness = self.sort(pop, fitness)
            pop[-1] = child_1; pop[-2] = child_2
            fitness, max_acc, avg_acc = self.fitness(pop,x,y)
            print('max acc generation - ',i+1,' : ',max_acc,' for max feat = ',self.max_feat)
            avg_acc_list.append(avg_acc)
            mses = [1/x for x in fitness]
            mse_list.append(np.average(mses))
            print(fitness, len(fitness))
            idxs = np.argmax(fitness)
            print(pop[idxs])
        idx_ = np.argmax(fitness)
        best_pop = pop[idx_]
        return mse_list, best_pop, max_acc, avg_acc_list

