# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:06:44 2023


"""
import pandas as pd
import numpy as np
import math

LD = pd.read_csv("C:/Users/yahan/Desktop/COMP565/A3/data/LD.csv")
beta = pd.read_csv("C:/Users/yahan/Desktop/COMP565/A3/data/beta_marginal.csv")
beta.rename(columns={'Unnamed: 0':'SNP'}, inplace=True)
SNPs=beta["SNP"].tolist()
beta = beta.set_index("SNP", drop = False)
del beta[beta.columns[0]]
LD.rename(columns={'Unnamed: 0':'SNP'}, inplace=True)
LD = LD.set_index("SNP", drop = False)
del LD[LD.columns[0]]
LD.values[[np.arange(LD.shape[0])]*2] = 1
#Q1 Expectation Step
#SNPs=beta["SNP"].tolist()
colnames=["SNP"]
VI_SNPs=pd.DataFrame(list(SNPs),columns=colnames)
VI_SNPs['posterior precision']=1
VI_SNPs['posterior mean']=0
VI_SNPs['PIP']=0.01
VI_SNPs = VI_SNPs.set_index("SNP", drop = False)
del VI_SNPs[VI_SNPs.columns[0]]
t_epsilon=1
t_beta=200
pi=0.01
N=439
M=100
ELBO_list=[]
for cycle in range(10):
 for i in SNPs:
    #update the posterior precison
    pos_pre=(N*t_epsilon)+t_beta
    VI_SNPs.loc[i,"posterior precision"]=pos_pre

    #update the posterior mean of the effect size
    b_i=beta.loc[i,"V1"]
    copy_SNPs=SNPs.copy()
    copy_SNPs.remove(i)
    total=0.0
    df_without_i=VI_SNPs.copy()
    df_without_i.drop(i,axis=0,inplace=True)
    pos_mean_term1=(df_without_i['PIP']*df_without_i['posterior mean']*LD[i]).sum()
   # for j in copy_SNPs:       
    #    r=LD.loc[i,j]
     #   mul= VI_SNPs.loc[j,'PIP']*VI_SNPs.loc[j,'posterior mean']*r
      #  print(mul)
       # total=total+mul
    
    pos_mean_beta=N*(t_epsilon/VI_SNPs.loc[i,"posterior precision"])*(b_i-pos_mean_term1)
    VI_SNPs.loc[i,"posterior mean"]=pos_mean_beta
    #update the PIP
    #first calculate the mean j
    mean_j=(math.log(pi/(1-pi)))+((1/2)*math.log(t_beta/VI_SNPs.loc[i,"posterior precision"]))+((VI_SNPs.loc[i,"posterior precision"]/2)*((VI_SNPs.loc[i,"posterior mean"])**2))
    pip=1/(1+math.exp(-(mean_j)))
    VI_SNPs.loc[i,"PIP"]=pip

    
    #if γj < 0.01 set it to 0.01, and if γj > 0.99 set it to 0.99.
 for i in SNPs:
    if(VI_SNPs.loc[i,"PIP"]>0.99):
        VI_SNPs.loc[i,"PIP"]=0.99
    elif(VI_SNPs.loc[i,"PIP"]<0.01):
        VI_SNPs.loc[i,"PIP"]=0.01
        
 #Q2 Maximization Step
 pi=(VI_SNPs["PIP"].sum())/M
 t_beta_term1=VI_SNPs['PIP']*(VI_SNPs['posterior mean'].to_numpy()**2+(1/(VI_SNPs['posterior precision'].to_numpy())))
 gamma=VI_SNPs['PIP'].sum()
 t_beta=1/(t_beta_term1.sum()/gamma)
        
 #Q3 Evidence lower bound

 #calculate the first term in ELBO
 term1=(N/2)*(math.log(t_epsilon))

 term2=(t_epsilon/2)*N
 
 term3_1=(VI_SNPs['PIP']*VI_SNPs['posterior mean']).to_numpy()
 term3=t_epsilon * np.matmul(np.transpose(term3_1), N*beta['V1'])

 term4_sum=(VI_SNPs['PIP']*(VI_SNPs['posterior mean']**2+(1/VI_SNPs['posterior precision']))).to_numpy()

 term4=N*(t_epsilon/2)*term4_sum.sum()

 list2=[]
 for a in SNPs:#from j=1 to M
   if(SNPs.index(a)<(len(SNPs)-1)):
    #k=SNPs.index(a)+1
    list1=[]
    #print(j)
    for k in range(SNPs.index(a)+1,len(SNPs)):#from k=j+1 to M
        product=VI_SNPs.loc[a,'PIP']*VI_SNPs.loc[a,'posterior mean']*VI_SNPs.loc[SNPs[k],'PIP']*VI_SNPs.loc[SNPs[k],'posterior mean']*(N*LD.loc[a,SNPs[k]])
        list1.append(product)
    list1_sum=sum(list1)
    list2.append(list1_sum)
 list2_sum=sum(list2)
 term5=list2_sum*t_epsilon

 ELBO_1=term1-term2+term3-term4-term5

 #calculate the second term in ELBO
 ELBO_2=(t_beta/2)*(term4_sum.sum())

 #calculate the third term in ELBO
 ELBO_3=-(1/2)*math.log(t_beta)*((VI_SNPs['PIP'].to_numpy()).sum())

 #calculate the fourth term in ELBO
 ELBO4_term1=math.log(pi)*VI_SNPs['PIP']+(1-VI_SNPs['PIP'])*math.log(1-pi)
 ELBO_4=ELBO4_term1.sum()

 #calculate the fifth term in ELBO
 ELBO5_term1=VI_SNPs['PIP']*np.log(VI_SNPs['PIP'])+(1-(VI_SNPs['PIP']))*np.log(1-VI_SNPs['PIP'])
 ELBO_5=ELBO5_term1.sum()

 #calculate the ELBO
 ELBO=ELBO_1+ELBO_2-ELBO_3+ELBO_4-ELBO_5
 ELBO_list.append(ELBO)

#Plot the scatter plot
import matplotlib.pyplot as plt
x=[0,1,2,3,4,5,6,7,8,9]
plt.scatter(x,ELBO_list)
#Q4 Evaluate PRS prediction
x_train=pd.read_csv("C:/Users/yahan/Desktop/COMP565/A3/data/X_train.csv")
y_train=pd.read_csv("C:/Users/yahan/Desktop/COMP565/A3/data/y_train.csv")
x_test=pd.read_csv("C:/Users/yahan/Desktop/COMP565/A3/data/X_test.csv")
y_test=pd.read_csv("C:/Users/yahan/Desktop/COMP565/A3/data/y_test.csv")

x_train =x_train.to_numpy()[:, 1:]
element_wise=VI_SNPs['PIP']*VI_SNPs['posterior mean']
element_wise=element_wise.to_numpy()
y_train_estimate=np.matmul(x_train,element_wise)

x_test=x_test.to_numpy()[:, 1:]
y_test_estimate=np.matmul(x_test,(VI_SNPs['PIP']*VI_SNPs['posterior mean']).to_numpy())

#Calculate pearson correlation coefficient
y_train_estimate=list(y_train_estimate)
from scipy.stats import pearsonr
corr, _ = pearsonr(y_train_estimate, y_train['V1'])
x=y_train_estimate
y=y_train['V1']
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
plt.scatter(x,y)
plt.plot(x, p(x), 'r')
plt.show()

x=list(y_test_estimate)
y=y_test['V1']
coefficients = np.polyfit(x, y, 1)
p = np.poly1d(coefficients)
plt.scatter(x,y)
plt.plot(x, p(x), 'r')
plt.show()
corr, _ = pearsonr(y_test_estimate, y_test['V1'])


#Q5 Evaluate fine-mapping
my_list = [i for i in range(0, 100)]
plt.scatter(my_list,list(VI_SNPs['PIP']))
plt.scatter(SNPs.index('rs9482449'),VI_SNPs.loc['rs9482449','PIP'],color='red')
plt.scatter(SNPs.index('rs7771989'),VI_SNPs.loc['rs7771989','PIP'],color='red')
plt.scatter(SNPs.index('rs2169092'),VI_SNPs.loc['rs2169092','PIP'],color='red')
plt.show()
