# -*- coding: utf-8 -*-
"""final project 565.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vCXzx_hBIMzLxEyKiHwhf5J1Xl_gkU4a
"""

pip install magenpy

"""# **Load the Data**"""

import magenpy as mgp
from google.colab import drive
drive.mount('/content/drive')
# Read the LD matrix:
ldm = mgp.LDMatrix.from_path("/content/drive/My Drive/565final/chr_22")
# print the number of SNPs:
print(ldm.n_snps)
# Convert to sparse matrix format:
cst_mat = ldm.to_csr_matrix()

a=ldm.snps

a

import pandas as pd
import numpy as np
LD_matrix_chr22=pd.DataFrame.sparse.from_spmatrix(cst_mat)

LD_matrix_chr22

columns=list(a)
print(columns)

LD_matrix_chr22.columns=columns

LD_matrix_chr22.index=LD_matrix_chr22.columns

LD_matrix_chr22

# Read the LD matrix:
ldm_2 = mgp.LDMatrix.from_path("/content/drive/My Drive/565final/chr_21")
# print the number of SNPs:
print(ldm_2.n_snps)
# Convert to sparse matrix format:
cst_mat_2 = ldm_2.to_csr_matrix()

annot= pd.read_csv("/content/drive/My Drive/565final/baselineLD.22.annot",delimiter='\t')
c=annot['SNP']

annot1= pd.read_csv("/content/drive/My Drive/565final/baselineLD.21.annot",delimiter='\t')
d=annot1['SNP']

annot_new=annot.select_dtypes(include=int)
annot_new['SNP']=c
annot_new=annot_new.drop(columns=['BP','CHR','base'])

annot_new1=annot1.select_dtypes(include=int)
annot_new1['SNP']=d
annot_new1=annot_new1.drop(columns=['BP','CHR','base'])

annot_new

annot_new['sum_ajk']=annot_new.iloc[:,0:96].sum(axis=1)
annot_new1['sum_ajk']=annot_new1.iloc[:,0:96].sum(axis=1)

annot_new

annot_new['SNP'][annot_new['Coding_UCSC'] == 1].tolist()
annot_new1['SNP'][annot_new1['Coding_UCSC'] == 1].tolist()

b=ldm_2.snps
LD_matrix_chr21=pd.DataFrame.sparse.from_spmatrix(cst_mat_2)
columns=list(b)
LD_matrix_chr21.columns=columns
LD_matrix_chr21.index=LD_matrix_chr21.columns

LD_matrix_chr21

train_sumstat1 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_1.sumstats.gz",delimiter='\t')

train_sumstat1_chrom21=train_sumstat1[train_sumstat1['#CHROM'] == 21]

train_sumstat2 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_2.sumstats.gz",delimiter='\t')

train_sumstat2_chrom22=train_sumstat2[train_sumstat2['#CHROM'] == 22]

train_sumstat2_chrom21=train_sumstat2[train_sumstat2['#CHROM'] == 21]

train_sumstat3 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_3.sumstats.gz",delimiter='\t')

train_sumstat3_chrom22=train_sumstat3[train_sumstat3['#CHROM'] == 22]

train_sumstat3_chrom21=train_sumstat3[train_sumstat3['#CHROM'] == 21]

train_sumstat4 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_4.sumstats.gz",delimiter='\t')

train_sumstat4_chrom22=train_sumstat4[train_sumstat4['#CHROM'] == 22]

train_sumstat4_chrom21=train_sumstat4[train_sumstat4['#CHROM'] == 21]

train_sumstat5 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_5.sumstats.gz",delimiter='\t')

train_sumstat5_chrom22=train_sumstat5[train_sumstat5['#CHROM'] == 22]
train_sumstat5_chrom22=train_sumstat5[train_sumstat5['#CHROM'] == 21]

"""# **Expectation Step**"""

import math
class VIPRS:
  def __init__(self,LD_matrix, annot_matrix):
    self.K=84 #96 annotations
    self.tau_epsilon=1
    self.tau_k=8400
    self.ks=annot_matrix.columns.tolist()
    self.ks=self.ks[0:84]
    self.w=pd.DataFrame(list(np.random.normal(0,0.001,84)),columns=["weight"])
    self.w['annotations']=self.ks
    self.w=self.w.set_index("annotations",drop=True)
    self.LD_matrix=LD_matrix
    self.SNPs=self.LD_matrix.columns.tolist()
    #self.SNPs=sum_stat['ID'].tolist()
    self.VI_SNPs=pd.DataFrame(self.SNPs,columns=["SNP"])
    self. VI_SNPs['posterior precision']=1
    self.VI_SNPs['posterior mean']=0
    self.VI_SNPs['PIP']=0.01
    self.VI_SNPs = self.VI_SNPs.set_index("SNP", drop = True)
    self.annot_matrix=annot_matrix
    annot=len(self.annot_matrix.columns)-2
    self.annot_copy=self.annot_matrix.iloc[:,0:annot]
    self.df_pi_j=pd.DataFrame(self.SNPs,columns=['SNP'])
    self.df_pi_j=self.df_pi_j.set_index("SNP",drop=True)
    self.df_pi_j['pi_j']=0
    self.annotations=self.annot_copy.columns.values.tolist()
    self.df_tau_k=pd.DataFrame(self.annotations,columns=['annotation'])
    self.df_tau_k=self.df_tau_k.set_index('annotation',drop=True)
    self.df_tau_k['tau_k']=0


  def expectation_step(self, sumstat_train):
    snps=0
    beta=sumstat_train
    SNPs=sumstat_train['ID'].tolist()
    for i in SNPs:

      N=sumstat_train['OBS_CT'][sumstat_train['ID']==i].values[0]
      #find the row that contains SNP i
      row_with_i = self.annot_matrix.index[self.annot_matrix['SNP'] == i].values
      row_with_i=row_with_i[0]


      #update the posterior precision tau_betaj
      pos_pre=(N*self.tau_epsilon)+self.annot_matrix.loc[row_with_i,'sum_ajk']

      self.VI_SNPs.loc[i,"posterior precision"]=pos_pre

      #update the posterior mean of the effect size
      b_i=beta.loc[i,"BETA"]

      copy_SNPs=SNPs.copy()
      copy_SNPs.remove(i)
      total=0.0
      df_without_i=self.VI_SNPs.copy()
      df_without_i.drop(i,axis=0,inplace=True)
      #LD-matrix
      pos_mean_term1=(df_without_i['PIP']*df_without_i['posterior mean']*self.LD_matrix[i]).sum()


      pos_mean_beta=N*(self.tau_epsilon/self.VI_SNPs.loc[i,"posterior precision"])*(b_i-pos_mean_term1)
      self.VI_SNPs.loc[i,"posterior mean"]=pos_mean_beta

      #
      pi_j_term1=-(self.annot_copy.iloc[row_with_i].values*self.w['weight'].values).sum()
      if pi_j_term1<-740:
        pi_j_term1=-740
      term2=math.exp(pi_j_term1)
      if term2<1e-6:
          term2=1e-6
      pi_j=1/(1+term2)

      mean_j=(math.log(pi_j/(1-pi_j)))+((1/2)*math.log(self.tau_k/self.VI_SNPs.loc[i,"posterior precision"]))+((self.VI_SNPs.loc[i,"posterior precision"]/2)*((self.VI_SNPs.loc[i,"posterior mean"])**2))
      pip=1/(1+math.exp(-(mean_j)))
      self.VI_SNPs.loc[i,"PIP"]=pip
     # print("hi")
      self.df_pi_j.loc[i,'pi_j']=pi_j
      snps+=1

      self.VI_SNPs[self.VI_SNPs <= 1e-6] = 1e-6
      self.VI_SNPs=self.VI_SNPs.fillna(0)
      self.df_pi_j[self.df_pi_j['pi_j']<=1e-6]=1e-6


  def maximization_step(self,lr,sumstat_train):
    self.df_pi_j
    SNPs=sumstat_train['ID'].tolist()
    #annotations=self.annot_copy.columns.values.tolist()
    #tau_k=pd.DataFrame(annotations,columns=['annotation'])
    #tau_k['tau_k']=0
    for k in self.annotations:
      #find the SNPs that equal to 1 in annotation k

      snps = self.annot_matrix['SNP'][self.annot_matrix[k] == 1].tolist()

      #df_snps=pd.DataFrame(snps,columns=['SNP'])

      #filter out the snps that are not in sum_stat

      common_snps = set(SNPs) & set(snps)
      df_snps=pd.DataFrame()

      df_snps['posterior precision']=self.VI_SNPs['posterior precision'].loc[common_snps]

      df_snps['posterior mean']=self.VI_SNPs['posterior mean'].loc[common_snps]

      df_snps['PIP']=self.VI_SNPs['PIP'].loc[common_snps]

      df_snps[df_snps['posterior precision'] <= 1e-6] = 1e-6
      df_snps[df_snps['PIP'] <= 1e-6] = 1e-6

      tao_k_term1=df_snps['PIP']*(df_snps['posterior mean'].to_numpy()**2+(1/(df_snps['posterior precision'].to_numpy())))

      gamma=df_snps['PIP'].sum()

      t_k=1/(tao_k_term1.sum()/gamma)

      self.df_tau_k.loc[k,'tau_k']=t_k


    self.tau_k=self.df_tau_k['tau_k'].sum()

    #keep only the SNPs that are in sum_stat
    mask = self.annot_matrix['SNP'].isin(self.SNPs)

    A=self.annot_matrix[mask]

    A=A.iloc[:,:-2]
    print(A)
    delta_w=np.matmul(A.transpose().to_numpy(),((self.VI_SNPs['PIP'].sub(self.df_pi_j['pi_j'])).to_numpy()))
    print(delta_w)

    self.w[self.w<1e-6]=1e-6

    self.w['weight']=(self.w['weight'].to_numpy()-(lr*delta_w))


  def VIPRS_iter(self,iter,lr,sumstat_train):
    for i in range(iter):
      print(i)
      self.expectation_step(sumstat_train)
      self.maximization_step(lr,sumstat_train)

  def evaluate(self,sum_stat_test,sum_stat_train):
    print('1')
    SNPs_test=sum_stat_test['SNP'].tolist()
    SNPs_train=sum_stat_train['ID'].tolist()
    print('2')
    common_snps = set(SNPs_test) & set(SNPs_train)
    print('3')
    common_snps=list(common_snps)
    print('4')
    #find the beta of test
    beta_test=sum_stat_test['BETA'].loc[common_snps].to_numpy()
    print('5')
    beta_hat=self.VI_SNPs['posterior mean'].loc[common_snps].to_numpy()
    #select the common snps in LD_matrix
    print('6')
    sub_LD=self.LD_matrix.loc[self.LD_matrix.index.isin(common_snps), common_snps]
    print('7')
    SS_res=1-np.matmul((2*np.transpose(beta_test)),beta_hat)+np.matmul((np.matmul(np.transpose(beta_hat),sub_LD.to_numpy())),beta_hat)

    return 1-(SS_res/1)

import math
math.exp(-740)

"""# Baseline Model"""

pip install viprs

import viprs as vp

import collections
collections.Iterable = collections.abc.Iterable

gdl21_f1=mgp.GWADataLoader(ld_store_files='/content/drive/My Drive/565final/chr_22',
                        sumstats_files="/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_1.sumstats.csv",
                        sumstats_format='plink')
gdl21_f2=mgp.GWADataLoader(ld_store_files='/content/drive/My Drive/565final/chr_22',
                        sumstats_files="/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_2.sumstats.csv",
                        sumstats_format='plink')

gdl21_f3=mgp.GWADataLoader(ld_store_files='/content/drive/My Drive/565final/chr_22',
                        sumstats_files="/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_3.sumstats.csv",
                        sumstats_format='plink')

gdl21_f4=mgp.GWADataLoader(ld_store_files='/content/drive/My Drive/565final/chr_22',
                        sumstats_files="/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_4.sumstats.csv",
                        sumstats_format='plink')

gdl21_f5=mgp.GWADataLoader(ld_store_files='/content/drive/My Drive/565final/chr_22',
                        sumstats_files="/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_5.sumstats.csv",
                        sumstats_format='plink')

gdl22_f1=mgp.GWADataLoader(ld_store_files='/content/drive/My Drive/565final/chr_22',
                        sumstats_files="/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_1.sumstats.csv",
                        sumstats_format='plink')
gdl22_f2=mgp.GWADataLoader(ld_store_files='/content/drive/My Drive/565final/chr_22',
                        sumstats_files="/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_2.sumstats.csv",
                        sumstats_format='plink')

gdl22_f3=mgp.GWADataLoader(ld_store_files='/content/drive/My Drive/565final/chr_22',
                        sumstats_files="/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_3.sumstats.csv",
                        sumstats_format='plink')

gdl22_f4=mgp.GWADataLoader(ld_store_files='/content/drive/My Drive/565final/chr_22',
                        sumstats_files="/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_4.sumstats.csv",
                        sumstats_format='plink')

gdl22_f5=mgp.GWADataLoader(ld_store_files='/content/drive/My Drive/565final/chr_22',
                        sumstats_files="/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_5.sumstats.csv",
                        sumstats_format='plink')

prs_list_22 = []
# Initialize VIPRS, passing it the GWADataLoader object
v1 = vp.VIPRS(gdl22_f1)
# Invoke the .fit() method to obtain posterior estimates
v1.fit()
test_prs_1 = v1.pseudo_validate(validation_gdl = gdl22_f1)
prs_list_22.append(test_prs_1)

v2 = vp.VIPRS(gdl22_f2)
v2.fit()
test_prs_2 = v2.pseudo_validate(validation_gdl = gdl22_f2)
prs_list_22.append(test_prs_2)

v3 = vp.VIPRS(gdl22_f3)
v3.fit()
test_prs_3 = v3.pseudo_validate(validation_gdl = gdl22_f3)
prs_list_22.append(test_prs_3)

v4 = vp.VIPRS(gdl22_f4)
v4.fit()
test_prs_4 = v4.pseudo_validate(validation_gdl = gdl22_f4)
prs_list_22.append(test_prs_4)

v5 = vp.VIPRS(gdl22_f5)
v5.fit()
test_prs_5 = v5.pseudo_validate(validation_gdl = gdl22_f5)
prs_list_22.append(test_prs_5)

prs_list_21 = []
# Initialize VIPRS, passing it the GWADataLoader object
v1 = vp.VIPRS(gdl21_f1)
# Invoke the .fit() method to obtain posterior estimates
v1.fit()
test_prs_1 = v1.pseudo_validate(validation_gdl = gdl21_f1)
prs_list_21.append(test_prs_1)

v2 = vp.VIPRS(gdl21_f2)
v2.fit()
test_prs_2 = v2.pseudo_validate(validation_gdl = gdl21_f2)
prs_list_21.append(test_prs_2)

v3 = vp.VIPRS(gdl21_f3)
v3.fit()
test_prs_3 = v3.pseudo_validate(validation_gdl = gdl21_f3)
prs_list_21.append(test_prs_3)

v4 = vp.VIPRS(gdl21_f4)
v4.fit()
test_prs_4 = v4.pseudo_validate(validation_gdl = gdl21_f4)
prs_list_21.append(test_prs_4)

v5 = vp.VIPRS(gdl21_f5)
v5.fit()
test_prs_5 = v5.pseudo_validate(validation_gdl = gdl21_f5)
prs_list_21.append(test_prs_5)

prs_list_22
#[0.08498457350162712,0.0871260128688872,0.08351989051187485,0.08609492667687163,0.08454158719816175]

prs_list_21
#[0.08503000568523839,0.08636922330480286,0.08356607111206008,0.08593283574074018,0.08452774447998919]#

"""#Chrom 21 Fold 1"""

train_sumstat1_chrom21=train_sumstat1[train_sumstat1['#CHROM'] == 21]
train_sumstat1_chrom21 = train_sumstat1_chrom21.set_index("ID", drop = False)

test_sumstat1 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/test/height_test_fold_1.csv.gz",delimiter='\t')
test_sumstat1_chrom21=test_sumstat1[test_sumstat1['CHR'] == 21]
test_sumstat1_chrom21 = test_sumstat1_chrom21.set_index("SNP", drop = False)

R_2_211=[]
for i in range(3):
  model.VIPRS_iter(1,0.01,train_sumstat1_chrom21)
  r=model.evaluate(test_sumstat1_chrom21,train_sumstat1_chrom21)
  R_2_211.append(r)

import matplotlib.pyplot as plt

plt.plot(R_2_211)
plt.xlabel('iterations')
plt.ylabel('R^2')
plt.title('R^2 estimate of Chrom 21 Fold 1')
plt.show()

R_211_max=max(R_2_211)
#R_211_max
R_2_211
R_2_211
#[0.004638390315765073, 0.03129437607875629, 0.028998197789681157]

fold1_21_bar=[]
fold1_21_bar.append(R_211_max)
fold1_21_bar.append(prs_list_21[0])
y_axis=['VIPRS_with_empirical prior','VIPRS']
plt.bar(y_axis, fold1_21_bar, color ='maroon',
        width = 0.4)

plt.xlabel("Models")
plt.ylabel("R_square")
plt.title("Model Comparison of Fold 1 Chrom 21")
plt.show()

"""#Chrom 21 Fold 2"""

train_sumstat2_chrom21=train_sumstat2[train_sumstat2['#CHROM'] == 21]
train_sumstat2_chrom21 = train_sumstat2_chrom21.set_index("ID", drop = False)

test_sumstat2 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/test/height_test_fold_2.csv.gz",delimiter='\t')
test_sumstat2_chrom21=test_sumstat2[test_sumstat2['CHR'] == 21]
test_sumstat2_chrom21 = test_sumstat2_chrom21.set_index("SNP", drop = False)

R_2_212=[]
for i in range(3):
  model.VIPRS_iter(1,0.01,train_sumstat2_chrom21)
  r=model.evaluate(test_sumstat2_chrom21,train_sumstat2_chrom21)
  R_2_212.append(r)

plt.plot(R_2_212)
plt.xlabel('iterations')
plt.ylabel('R^2')
plt.title('R^2 estimate of Chrom 21 Fold 2')
plt.show()

R_212_max=max(R_2_212)
R_212_max
#0.0865451955782438
R_2_212
#[0.0865451955782438, 0.06724703604585935, 0.059427955258247955]

fold2_21_bar=[]
fold2_21_bar.append(R_212_max)
fold2_21_bar.append(prs_list_21[1])
y_axis=['VIPRS_with_empirical prior','VIPRS']
plt.bar(y_axis, fold2_21_bar, color ='maroon',
        width = 0.4)

plt.xlabel("Models")
plt.ylabel("R_square")
plt.title("Model Comparison of Fold 2 Chrom 21")
plt.show()

"""#Chrom 21 Fold 3"""

train_sumstat3_chrom21=train_sumstat3[train_sumstat3['#CHROM'] == 21]
train_sumstat3_chrom21 = train_sumstat3_chrom21.set_index("ID", drop = False)

test_sumstat3 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/test/height_test_fold_3.csv.gz",delimiter='\t')
test_sumstat3_chrom21=test_sumstat3[test_sumstat3['CHR'] == 21]
test_sumstat3_chrom21 = test_sumstat3_chrom21.set_index("SNP", drop = False)

R_2_213=[]
for i in range(3):
  model.VIPRS_iter(1,0.01,train_sumstat3_chrom21)
  r=model.evaluate(test_sumstat3_chrom21,train_sumstat3_chrom21)
  R_2_213.append(r)

plt.plot(R_2_213)
plt.xlabel('iterations')
plt.ylabel('R^2')
plt.title('R^2 estimate of Chrom 21 Fold 3')
plt.show()

R_213_max=max(R_2_213)
R_213_max
#0.0652531168171745

fold3_21_bar=[]
fold3_21_bar.append(R_213_max)
fold3_21_bar.append(prs_list_21[2])
y_axis=['VIPRS_with_empirical prior','VIPRS']
plt.bar(y_axis, fold3_21_bar, color ='maroon',
        width = 0.4)

plt.xlabel("Models")
plt.ylabel("R_square")
plt.title("Model Comparison of Fold 3 Chrom 21")
plt.show()

"""#Chrom 21 Fold 4"""

train_sumstat4_chrom21=train_sumstat4[train_sumstat4['#CHROM'] == 21]
train_sumstat4_chrom21 = train_sumstat4_chrom21.set_index("ID", drop = False)

test_sumstat4 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/test/height_test_fold_4.csv.gz",delimiter='\t')
test_sumstat4_chrom21=test_sumstat4[test_sumstat4['CHR'] == 21]
test_sumstat4_chrom21 = test_sumstat4_chrom21.set_index("SNP", drop = False)

model=VIPRS(LD_matrix_chr21,annot_new1)

R_2_214=[]
for i in range(3):
  model.VIPRS_iter(1,0.01,train_sumstat4_chrom21)
  r=model.evaluate(test_sumstat4_chrom21,train_sumstat4_chrom21)
  R_2_214.append(r)

plt.plot(R_2_214)
plt.xlabel('iterations')
plt.ylabel('R^2')
plt.title('R^2 estimate of Chrom 21 Fold 4')
plt.show()

R_214_max=max(R_2_214)
R_214_max
#0.033124136328837483

fold4_21_bar=[]
fold4_21_bar.append(R_214_max)
fold4_21_bar.append(prs_list_21[3])
y_axis=['VIPRS_with_empirical prior','VIPRS']
plt.bar(y_axis, fold4_21_bar, color ='maroon',
        width = 0.4)

plt.xlabel("Models")
plt.ylabel("R_square")
plt.title("Model Comparison of Fold 4 Chrom 21")
plt.show()

"""#Chrom 21 Fold 5"""

train_sumstat5_chrom21=train_sumstat5[train_sumstat5['#CHROM'] == 21]
train_sumstat5_chrom21 = train_sumstat5_chrom21.set_index("ID", drop = False)

test_sumstat5 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/test/height_test_fold_5.csv.gz",delimiter='\t')
test_sumstat5_chrom21=test_sumstat5[test_sumstat5['CHR'] == 21]
test_sumstat5_chrom21 = test_sumstat5_chrom21.set_index("SNP", drop = False)

model=VIPRS(LD_matrix_chr21,annot_new1)

R_2_215=[]
for i in range(3):
  model.VIPRS_iter(1,0.01,train_sumstat5_chrom21)
  r=model.evaluate(test_sumstat5_chrom21,train_sumstat5_chrom21)
  R_2_215.append(r)

plt.plot(R_2_215)
plt.xlabel('iterations')
plt.ylabel('R^2')
plt.title('R^2 estimate of Chrom 21 Fold 5')
plt.show()

R_215_max=max(R_2_215)
R_215_max
#0.04441778688496456

fold5_21_bar=[]
fold5_21_bar.append(R_215_max)
fold5_21_bar.append(prs_list_21[4])
y_axis=['VIPRS_with_empirical prior','VIPRS']
plt.bar(y_axis, fold5_21_bar, color ='maroon',
        width = 0.4)

plt.xlabel("Models")
plt.ylabel("R_square")
plt.title("Model Comparison of Fold 5 Chrom 21")
plt.show()

"""# **Chrom 22 Fold 1**"""

train_sumstat1 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_1.sumstats.gz",delimiter='\t')

train_sumstat1_chrom22=train_sumstat1[train_sumstat1['#CHROM'] == 22]
train_sumstat1_chrom22 = train_sumstat1_chrom22.set_index("ID", drop = False)

train_sumstat1_chrom22

test_sumstat1 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/test/height_test_fold_1.csv.gz",delimiter='\t')
test_sumstat1_chrom22=test_sumstat1[test_sumstat1['CHR'] == 22]
test_sumstat1_chrom22 = test_sumstat1_chrom22.set_index("SNP", drop = False)
test_sumstat1_chrom22

model=VIPRS(LD_matrix_chr22,annot_new)

R_2=[]
for i in range(3):
  model.VIPRS_iter(1,0.01,train_sumstat1_chrom22)
  r=model.evaluate(test_sumstat1_chrom22,train_sumstat1_chrom22)
  R_2.append(r)

R_2

import matplotlib.pyplot as plt
plt.plot(R_2)
plt.xlabel('iterations')
plt.ylabel('R^2')
plt.title('R^2 estimate of Chrom 22 Fold 1')
plt.show()

R_2_max=max(R_2)
R_2_max
#0.026147410335074506

fold1_22_bar=[]
fold1_22_bar.append(R_2_max)
fold1_22_bar.append(prs_list_22[0])
y_axis=['VIPRS_with_empirical prior','VIPRS']
plt.bar(y_axis, fold1_22_bar, color ='maroon',
        width = 0.4)

plt.xlabel("Models")
plt.ylabel("R_square")
plt.title("Model Comparison of Fold 2 Chrom 22")
plt.show()

"""## Chrom 22 Fold 2"""

#perform the second fold cross-validation
train_sumstat2 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_2.sumstats.gz",delimiter='\t')
train_sumstat2_chrom22=train_sumstat2[train_sumstat2['#CHROM'] == 22]
train_sumstat2_chrom22 = train_sumstat2_chrom22.set_index("ID", drop = False)

train_sumstat1_chrom22

test_sumstat2 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/test/height_test_fold_2.csv.gz",delimiter='\t')
test_sumstat2_chrom22=test_sumstat2[test_sumstat2['CHR'] == 22]
test_sumstat2_chrom22 = test_sumstat2_chrom22.set_index("SNP", drop = False)

R_2_fold2=[]
for i in range(3):
  print('iter'+str(i))
  model.VIPRS_iter(1,0.01,train_sumstat2_chrom22)
  r=model.evaluate(test_sumstat2_chrom22,train_sumstat2_chrom22)
  R_2_fold2.append(r)

R_2_fold2

import matplotlib.pyplot as plt
plt.plot(R_2_fold2)
plt.xlabel('iterations')
plt.ylabel('R^2')
plt.title('R^2 estimate of Chrom 22 Fold 2')
plt.show()

R_2_max=max(R_2_fold2)
R_2_max
#0.09635496600025073

fold2_22_bar=[]
fold2_22_bar.append(R_2_max)
fold2_22_bar.append(prs_list_22[1])
y_axis=['VIPRS_with_empirical prior','VIPRS']
plt.bar(y_axis, fold2_22_bar, color ='maroon',
        width = 0.4)

plt.xlabel("Models")
plt.ylabel("R_square")
plt.title("Model Comparison of Fold 1 Chrom 22")
plt.show()

"""# Chrom 22 Fold 3"""

#perform the second fold cross-validation
train_sumstat3 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_3.sumstats.gz",delimiter='\t')
train_sumstat3_chrom22=train_sumstat3[train_sumstat3['#CHROM'] == 22]
train_sumstat3_chrom22 = train_sumstat3_chrom22.set_index("ID", drop = False)

test_sumstat3 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/test/height_test_fold_3.csv.gz",delimiter='\t')
test_sumstat3_chrom22=test_sumstat3[test_sumstat3['CHR'] == 22]
test_sumstat3_chrom22 = test_sumstat3_chrom22.set_index("SNP", drop = False)
test_sumstat3_chrom22

R_2_fold3=[]
for i in range(3):
  print('iter'+str(i))
  model.VIPRS_iter(1,0.01,train_sumstat3_chrom22)
  r=model.evaluate(test_sumstat3_chrom22,train_sumstat3_chrom22)
  R_2_fold3.append(r)

R_23_max=max(R_2_fold3)
R_23_max
#0.08778747598213366



R_2_fold3

plt.plot(R_2_fold3)
plt.xlabel('iterations')
plt.ylabel('R^2')
plt.title('R^2 estimate of Chrom 22 Fold 3')
plt.show()

fold3_22_bar=[]
fold3_22_bar.append(R_23_max)
fold3_22_bar.append(prs_list_22[2])
y_axis=['VIPRS_with_empirical prior','VIPRS']
plt.bar(y_axis, fold3_22_bar, color ='maroon',
        width = 0.4)

plt.xlabel("Models")
plt.ylabel("R_square")
plt.title("Model Comparison of Fold 3 Chrom 22")
plt.show()

prs_list_22

"""# Chrom 22 Fold 4"""

#perform the second fold cross-validation
train_sumstat4 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_4.sumstats.gz",delimiter='\t')
train_sumstat4_chrom22=train_sumstat4[train_sumstat4['#CHROM'] == 22]
train_sumstat4_chrom22 = train_sumstat4_chrom22.set_index("ID", drop = False)

test_sumstat4 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/test/height_test_fold_4.csv.gz",delimiter='\t')
test_sumstat4_chrom22=test_sumstat4[test_sumstat4['CHR'] == 22]
test_sumstat4_chrom22 = test_sumstat4_chrom22.set_index("SNP", drop = False)
test_sumstat4_chrom22

R_2_fold4=[]
for i in range(3):
  print('iter'+str(i))
  model.VIPRS_iter(1,0.01,train_sumstat4_chrom22)
  r=model.evaluate(test_sumstat4_chrom22,train_sumstat4_chrom22)
  R_2_fold4.append(r)

R_24_max=max(R_2_fold4)
R_24_max
#0.0561943050830247

plt.plot(R_2_fold4)
plt.xlabel('iterations')
plt.ylabel('R^2')
plt.title('R^2 estimate of Chrom 22 Fold 4')
plt.show()

fold4_22_bar=[]
fold4_22_bar.append(R_24_max)
fold4_22_bar.append(prs_list_22[3])
y_axis=['VIPRS_with_empirical prior','VIPRS']
plt.bar(y_axis, fold4_22_bar, color ='maroon',
        width = 0.4)

plt.xlabel("Models")
plt.ylabel("R_square")
plt.title("Model Comparison of Fold 4 Chrom 22")
plt.show()

"""#Chrom 22 Fold 5"""

#perform the second fold cross-validation
train_sumstat5 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/training/height_fold_5.sumstats.gz",delimiter='\t')
train_sumstat5_chrom22=train_sumstat5[train_sumstat5['#CHROM'] == 22]
train_sumstat5_chrom22 = train_sumstat5_chrom22.set_index("ID", drop = False)

test_sumstat5 = pd.read_csv("/content/drive/My Drive/565final/data 3/sumstats/height/test/height_test_fold_5.csv.gz",delimiter='\t')
test_sumstat5_chrom22=test_sumstat5[test_sumstat5['CHR'] == 22]
test_sumstat5_chrom22 = test_sumstat5_chrom22.set_index("SNP", drop = False)
test_sumstat5_chrom22

R_2_fold5=[]
for i in range(3):
  print('iter'+str(i))
  model.VIPRS_iter(1,0.01,train_sumstat5_chrom22)
  r=model.evaluate(test_sumstat5_chrom22,train_sumstat5_chrom22)
  R_2_fold5.append(r)

R_25_max=max(R_2_fold5)
R_25_max
#0.095155256693777

plt.plot(R_2_fold5)
plt.xlabel('iterations')
plt.ylabel('R^2')
plt.title('R^2 estimate of Chrom 22 Fold 5')
plt.show()

fold5_22_bar=[]
fold5_22_bar.append(R_25_max)
fold5_22_bar.append(prs_list_22[4])
y_axis=['VIPRS_with_empirical prior','VIPRS']
plt.bar(y_axis, fold5_22_bar, color ='maroon',
        width = 0.4)

plt.xlabel("Models")
plt.ylabel("R_square")
plt.title("Model Comparison of Fold 5 Chrom 22")
plt.show()