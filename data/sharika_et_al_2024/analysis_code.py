# Import packages for analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import csv
from SMdRQA.RP_maker import RP_computer
from SMdRQA.Extract_from_RP import windowed_RP
from SMdRQA.Extract_from_RP import First_middle_last_sliding_windows_all_vars
from SMdRQA.cross_validation import nested_cv

# Define function for reading .xlsx files and to give the HRV values as a numpy array
# For details about data and questionnaire scores see the file : https://github.com/SwaragThaikkandi/SMdRQA/blob/main/data/grpInfo-2-20.11.2022.xlsx


def read_HR(path):
    df=pd.read_excel(path, index_col=0)
    df.columns=df.iloc[1]
    df=df[2:]
    HR=df['HR (bpm)'].to_numpy()
    return HR
# Load Outcome data
outcome_data = pd.read_csv('/home/swarag0/Group_study/Grp_outcome.csv')
GroupID = np.array(outcome_data['Grp ID'])
outcome = np.array(outcome_data['outcome\t'])

# Checking the number of data files

dir='/home/swarag0/Group_study/Polar-GrpStudyHR-Data' # replace this part with the directory where you have saved the datafiles(only the data files)
files=[f for f in listdir(dir) if isfile(join(dir, f))]
if len(files)!=489:                                   # Total number of data files in our experiment was 489, this line of code was to confirm that we didn't miss any
   print("File count didn't match")

# Extract data from xlsx files into a dictionary

extracted_data=defaultdict(lambda:defaultdict(lambda:defaultdict()))
for file in files:
    l=file.strip().split('-')
    gd=l[0]
    grp=int(l[1])
    #mem=int(str(l[2])[:2])
    mem=int(l[2].split('_')[0])
    extracted_data[gd][grp][mem]=read_HR(join(dir,file))
    #extracted_data[gd][grp][mem]=read_HR_rn(join(dir,file))

Groups = [2,5,6,7,8,11,12,14,15,16,17,19,20,21,22,23,24,25,26,30,31,32,33,34,35,36,37,38,39,40,41,42,43,45,46,49,51,52,53,54,55,56,57,59]  # Groups 44 and 50 were eliminated due to shorter time series length(<2min)

data_types = ['GD_THS','preGD_THS']

# Generate numpy files for preGD data, we will only take 300s duration

for grp in Groups:
  d=int(len(extracted_data['preGD_THS'][grp]))  # determining the number of members in the group
  if grp == 30:
    data = np.zeros((115,d))
    data_rn = np.zeros((150,d))
    for member in range(d):
      array = extracted_data['preGD_THS'][grp][member+1] # we need to do this because python indexing starts from zero whereas the assigned numbers of group members started from 1
      data[:,member] = array[10:125]
      data_rn[:,member] = np.random.permutation(array[10:125]) # randomized data

    np.save('/home/swarag0/Group_study/data/(preGD,'+str(grp)+','+str(outcome[GroupID == grp][0])+').npy',data) # save data with name (type of data, group, outcome)
    np.save('/home/swarag0/Group_study/data/(rn_preGD,'+str(grp)+','+str(outcome[GroupID == grp][0])+').npy',data_rn)
    
  elif grp == 24:
    data = np.zeros((150,d))
    data_rn = np.zeros((150,d))
    for member in range(d):
      array = extracted_data['preGD_THS'][grp][member+1]
      data[:,member] = array[50:200]
      data_rn[:,member] = np.random.permutation(array[50:200]) # randomized data

    np.save('/home/swarag0/Group_study/data/(preGD,'+str(grp)+','+str(outcome[GroupID == grp][0])+').npy',data) # save data with name (type of data, group, outcome)
    np.save('/home/swarag0/Group_study/data/(rn_preGD,'+str(grp)+','+str(outcome[GroupID == grp][0])+').npy',data_rn)

  else:
    data = np.zeros((150,d))
    data_rn = np.zeros((150,d))
    for member in range(d):
      array = extracted_data['preGD_THS'][grp][member+1]
      data[:,member] = array[0:150]
      data_rn[:,member] = np.random.permutation(array[10:150]) # randomized data

    np.save('/home/swarag0/Group_study/data/(preGD,'+str(grp)+','+str(outcome[GroupID == grp][0])+').npy',data) # save data with name (type of data, group, outcome)
    np.save('/home/swarag0/Group_study/data/(rn_preGD,'+str(grp)+','+str(outcome[GroupID == grp][0])+').npy',data_rn)


# Generate numpy files for GD data, we will only take 300s duration

for grp in Groups:
  d=int(len(extracted_data['GD_THS'][grp]))  # determining the number of members in the group
  data = np.zeros((115,d))
  data_rn = np.zeros((150,d))
  for member in range(d):
    array = extracted_data['GD_THS'][grp][member+1] # we need to do this because python indexing starts from zero whereas the assigned numbers of group members started from 1
    data[:,member] = array
    data_rn[:,member] = np.random.permutation(array) # randomized data

  np.save('/home/swarag0/Group_study/data/(GD,'+str(grp)+','+str(outcome[GroupID == grp][0])+').npy',data) # save data with name (type of data, group, outcome)
  np.save('/home/swarag0/Group_study/data/(rn_GD,'+str(grp)+','+str(outcome[GroupID == grp][0])+').npy',data_rn)
    

##################################################################### IMPORTANT #######################################################################################
# The following details are added for giving a complete picture, in some cases paramteter exploration would take multiple runs and this code does not intend to give
# a perception that the entire analysis could be done by running a single script.  

# Generate RPs for the files saved in the data folder
input_path = '/home/swarag0/Group_study/data'
RP_dir = '/home/swarag0/Group_study/RP'
RP_computer(input_path, RP_dir)

## Extract RQA variables 

Dict_RPs = windowed_RP(68, '/home/swarag0/Group_study/RP', '/home/swarag0/Group_study')

First_middle_last_sliding_windows_all_vars(Dict_RPs, '/home/swarag0/Group_study/Group_RQA_data.csv')

## Extract Information 

RQA_data = pd.read_csv('/home/swarag0/Group_study/Group_RQA_data.csv')

Info = np.array(RQA_data['group'])

TYPE = []
GROUP = []
OUTCOME = []

for i in range(len(Info)):
  Type, group, outcome = eval(Info[i])
  TYPE.append(Type)
  GROUP.append(group)
  OUTCOME.append(outcome)

RQA_data['group'] = GROUP
RQA_data['type'] = TYPE
RQA_data['outcome'] = OUTCOME

## example comparison:  Between GR and random GD, and selecting mode of sliding window distribution

GD_data_1 = RQA_data[RQA_data['type'] == 'GD'].reset_index(drop = True)
GD_data = GD_data_1[GD_data_1['window'] == 'mode'].reset_index(drop = True)

rn_GD_data_1 = RQA_data[RQA_data['type'] == 'rn_GD'].reset_index(drop = True)
rn_GD_data = rn_GD_data_1[rn_GD_data_1['window'] == 'mode'].reset_index(drop = True)

# IMPORTANT: Both the data should have the same ordering if we are planning to use pair-wise comparison for outerloop validation accuracies

sorted_GD_data = GD_data.sort_values(by='group')
sorted_rn_GD_data = rn_GD_data.sort_values(by='group')

# nested cross validation

features=['recc_rate',
 'percent_det',
 'avg_diag',
 'max_diag',
 'percent_lam',
 'avg_vert',
 'vert_ent',
 'diag_ent',
 'vert_max']

 nested_cv(sorted_GD_data, features, 'outcome', 'GD_mode', repeats=200, inner_repeats=20, outer_splits=3, inner_splits=2)
 nested_cv(sorted_rn_GD_data, features, 'outcome', 'rn_GD_mode', repeats=200, inner_repeats=20, outer_splits=3, inner_splits=2)

 # Now we can do comparison 
