import numpy as np
import pandas as pd
import os
import sys
#DataPath = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","data_tukiyama","SSBCI","SSBCI","recorded_EEGs","maeda","CSV")

#data = pd.read_csv(os.path.join(DataPath,"data_1","overt_1.CSV"))

#print(data.head())

#StimulusPath = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","data_tukiyama","SSBCI","SSBCI","stimulus")
StimulusPath = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","data_tukiyama","SSBCI","SSBCI","stimulus")
#Save_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","covert","div_data")
Save_Path = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","1_word_HMM","data","covert","div_data")

#stimulusData = pd.read_excel(os.path.join(StimulusPath,"stimulus_1.xlsx"))

#print(stimulusData.head())
#print(stimulusData["covert"])
#data = data.iloc[:,1:]

count = 0
flag = 0
data_size = 0
start = 0
finish = 0
segment_size = 1024
Dataset_num = 60
sub_names = ["maeda","shirai","sunaba","takahashi"]
#ch_names = [" F7-REF"," T7-REF"," Cz-REF"]


ch_names = [" F7-REF"," T7-REF"," P7-REF"," O1-REF"," O2-REF"," P8-REF"," T8-REF"," F8-REF"," Fp2-REF"," Fp1-REF"," F3-REF"," C5-REF"," P3-REF"," P4-REF"," C6-REF"," F4-REF"," Fz-REF"," Cz-REF"," Pz-REF"]


for ch_name in ch_names:
	 
	data_a = np.empty((0,segment_size),float)
	data_i = np.empty((0,segment_size),float)
	data_u = np.empty((0,segment_size),float) 
	data_e = np.empty((0,segment_size),float)
	data_o = np.empty((0,segment_size),float)

	for set_num in range(Dataset_num):
		stimulusData = pd.read_excel(os.path.join(StimulusPath,"stimulus_"+str(set_num+1)+".xlsx"))
		for sub_name in sub_names:
			#DataPath = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","data_tukiyama","SSBCI","SSBCI","recorded_EEGs",sub_name,"CSV")
			DataPath = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","data_tukiyama","SSBCI","SSBCI","recorded_EEGs",sub_name,"CSV")

		 
			data = pd.read_csv(os.path.join(DataPath,"data_"+str(set_num+1),"covert_"+str(set_num+1)+".CSV"))
			
			dev_data = np.empty((0,segment_size),float)
			for i in range(len(data)-1):
				if(data.loc[i," EXT"]>10000 and flag ==0 ):
					flag=1
					count+=1  
					start=i

				elif(flag==1 and data.loc[i," EXT"]<10000):
					flag=0
					finish = i
					print(finish-start)
					sys.exit()
					dev_data= np.vstack([dev_data ,data.loc[start:start+segment_size-1,ch_name].values])


			print(count)

			stimulus_num = stimulusData["covert"].str[0].astype(int)

			for i in range(len(stimulusData)):
				if(stimulus_num[i]==1):
					data_a = np.vstack([data_a,dev_data[i]]) 
				elif(stimulus_num[i]==2):
					data_i = np.vstack([data_i,dev_data[i]]) 
				elif(stimulus_num[i]==3):
					data_u = np.vstack([data_u,dev_data[i]]) 
				elif(stimulus_num[i]==4):
					data_e = np.vstack([data_e,dev_data[i]]) 
				elif(stimulus_num[i]==5):
					data_o = np.vstack([data_o,dev_data[i]]) 

	print(data_a.shape)
	print(data_i.shape)
	print(data_u.shape)
	print(data_e.shape)
	print(data_o.shape)
	np.savetxt(os.path.join(Save_Path,"data_a"+str(ch_name)+".csv"),data_a,delimiter=',')
	np.savetxt(os.path.join(Save_Path,"data_i"+str(ch_name)+".csv"),data_i,delimiter=',')
	np.savetxt(os.path.join(Save_Path,"data_u"+str(ch_name)+".csv"),data_u,delimiter=',')
	np.savetxt(os.path.join(Save_Path,"data_e"+str(ch_name)+".csv"),data_e,delimiter=',')
	np.savetxt(os.path.join(Save_Path,"data_o"+str(ch_name)+".csv"),data_o,delimiter=',')


print(data_a)
