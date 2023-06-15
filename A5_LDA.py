# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:07:05 2023


"""

#NAME: Yahan Zhang
#Student ID: 260908840
import pandas as pd
import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt
import seaborn as sns
df_mimics3 = pd.read_csv("./MIMIC3_DIAGNOSES_ICD_subset.csv")
df_mimics3['topic 1']=0
df_mimics3['topic 2']=0
df_mimics3['topic 3']=0
df_mimics3['topic 4']=0
df_mimics3['topic 5']=0

#construct a patients x topics dataframe
patients=df_mimics3['SUBJECT_ID'].unique()
df_DK=pd.DataFrame(patients, columns=['patients'])
df_DK = df_DK.set_index('patients', drop = True)
df_DK['topic 1']=0
df_DK['topic 2']=0
df_DK['topic 3']=0
df_DK['topic 4']=0
df_DK['topic 5']=0
#construct a ICD-codes x topics dataframe
ICD_codes=df_mimics3['ICD9_CODE'].unique()
df_WK=pd.DataFrame(ICD_codes,columns=['ICD'])
df_WK=df_WK.set_index('ICD',drop=True)
df_WK['topic 1']=0
df_WK['topic 2']=0
df_WK['topic 3']=0
df_WK['topic 4']=0
df_WK['topic 5']=0

M=389
D=689
alpha=1
beta=0.001
K=5
#Zid is one-hot encoded
for j in range(10):
 print(j)
 
 for i, row in df_mimics3.iterrows():
    #find the current patient
    patient=df_mimics3.iloc[i,0]
    #find the current ICD_code
    ICD=df_mimics3.iloc[i,1]
    #update the topic distribution    
    gamma_idk=np.zeros(5)
  
    for topic in range(1,6):
       topic_name='topic '+str(topic)
       col_number=(topic-1)+2
       
        
       n_dk=df_DK.loc[patient,topic_name]-df_mimics3.iloc[i,col_number]
       n_wk_sum=df_WK[topic_name].sum()-df_mimics3.iloc[i,col_number]
       n=df_WK.loc[ICD,topic_name]-df_mimics3.iloc[i,col_number]
       
        #calculate gamma idk
       gamma_term1=alpha+n_dk
      
       gamma_term2=(beta+n)/(M*beta+n_wk_sum)
       gamma_k=gamma_term1*gamma_term2
       gamma_idk[topic-1]=gamma_k
       
    gamma_idk = gamma_idk / gamma_idk.sum()

    z_id=np.random.multinomial(1,gamma_idk,size=1)
    z_id=z_id.tolist()
    #update the z_id to the original file
    #find the index when z_id=1
    index_of_one = z_id[0].index(1)
    topic='topic '+str(index_of_one+1)
    df_mimics3.loc[i,topic]=1
    #update the sufficient statistics
    df_DK.loc[patient,topic]+=1
    df_WK.loc[ICD,topic]+=1
 
 
    
 

#normalize the final ICDs-by-topics and the patients-by-topics matrix
phi_WK=pd.DataFrame(ICD_codes,columns=['ICD'])
phi_WK=phi_WK.set_index('ICD',drop=True)
phi_WK['topic 1']=0
phi_WK['topic 2']=0
phi_WK['topic 3']=0
phi_WK['topic 4']=0
phi_WK['topic 5']=0

for i in range(phi_WK.shape[0]):
    ICD = phi_WK.index[i]
    for k in range(1,6):
        topic='topic '+str(k)
        n_wk=df_WK.loc[ICD,topic]
        sum_n_wk=df_WK[topic].sum()
        phi_WK.loc[ICD,topic]=(beta+n_wk)/((M*beta)+sum_n_wk)
        
theta_DK=pd.DataFrame(patients,columns=['patients'])
theta_DK=theta_DK.set_index('patients',drop=True)
theta_DK['topic 1']=0
theta_DK['topic 2']=0
theta_DK['topic 3']=0
theta_DK['topic 4']=0
theta_DK['topic 5']=0

for i in range(theta_DK.shape[0]):
    patient = theta_DK.index[i]
    for k in range(1,6):
        topic='topic '+str(k)
        n_dk=df_DK.loc[patient,topic]
        sum_n_dk=df_DK.iloc[i].sum()
        theta_DK.loc[patient,topic]=(alpha+n_dk)/((K*alpha)+sum_n_dk)
        

#visualize the top ICD codes under each topic
topics_ICD_matrix=pd.DataFrame()
for i in range(1,6):
    topic='topic '+str(i)
    top5 = phi_WK.nlargest(10,topic)
    topics_ICD_matrix = topics_ICD_matrix.append(top5)
#find the corresponding disease
df_ICD_titles = pd.read_csv("./D_ICD_DIAGNOSES_DATA_TABLE.csv")

for i in range(topics_ICD_matrix.shape[0]):
    print(i)
    current_ICD=topics_ICD_matrix.index[i]
    try:
      idx = df_ICD_titles.index[df_ICD_titles['ICD9_CODE'] == str(current_ICD)].tolist()[0]
      title = df_ICD_titles.loc[idx, 'SHORT_TITLE']
      concatenate = str(current_ICD) + "-" + str(title)
      topics_ICD_matrix.rename(index={current_ICD: concatenate}, inplace=True)
    except IndexError:
      print("ICD code not found in df_ICD_titles:", current_ICD)
fig, ax = plt.subplots(figsize=(10, 20))
ICD_topics = heatmap(topics_ICD_matrix, cmap="Reds", linecolor="white")

#correlate topics with the target ICD codes

#construct a matrix as patientsx3
patient_3icd=pd.DataFrame(patients,columns=['patients'])
patient_3icd=patient_3icd.set_index('patients',drop=True)
patient_3icd['331']=0
patient_3icd['332']=0
patient_3icd['340']=0


for patient in patients:
    #find the related ICD code of this patient
    ICD_codes=df_mimics3.loc[df_mimics3['SUBJECT_ID'] == patient , 'ICD9_CODE']
    ICD_codes=ICD_codes.tolist()
    #check if they contain code starts with 331,332 or 340
    for code in ICD_codes:
        first_three_ints = str(code)[:3]
        if(first_three_ints=='331'):
            patient_3icd.loc[patient,'331']=1
        if(first_three_ints=='332'):
            patient_3icd.loc[patient,'332']=1
        if(first_three_ints=='340'):
             patient_3icd.loc[patient,'340']=1
             
# merge the two dataframes on PatientID
merged_df = pd.merge(theta_DK, patient_3icd, on='patients')

#creat topics by ICD-codes matrix
topics_ICD=pd.DataFrame()
topics_ICD['331']=0
topics_ICD['332']=0
topics_ICD['340']=0
ICD=['331','332','340']
topics=['topic 1','topic 2','topic 3','topic 4','topic 5']
for icd in ICD:
    for topic in topics:
        topics_ICD.loc[topic,icd]=merged_df[icd].corr(merged_df[topic])


fig_Q3, ax_Q3 = plt.subplots(figsize=(5, 3))
ax_Q3 = heatmap(topics_ICD, cmap="Reds", linecolor="white")
    


#Generate the heatmap for the top 100 patients
topics_patients_matrix=pd.DataFrame()
for i in range(1,6):
    topic='topic '+str(i)
    top100 = theta_DK.nlargest(100,topic)
    top100 = top100.fillna(0)
    topics_patients_matrix = topics_patients_matrix.append(top100)
topics_patients_matrix.fillna(0)            
fig_Q4, ax_Q4 = plt.subplots(1,figsize=(10, 30))
ht_Q4= heatmap(topics_patients_matrix, cmap="Reds", linecolor="white")
     
     
        
    
        
        
