import csv
import pandas as pd
import numpy as np
import os
import shutil
from glob import glob
import math
#TSFRESH
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#ROC
import numpy as np   
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score 

def rawToRecord():
	files=["combine.csv"] # **Modify List of Input Files:** part1.csv, part2.csv
	for filename in files:
		file = pd.read_csv(filename, error_bad_lines=False)

		for i, x in file.groupby('bookingID'):
			if os.path.exists(str(i).join(".csv")):
				x.to_csv('{}.csv'.format(i), index=False, mode='a') # append if already exists
			else:
				x.to_csv('{}.csv'.format(i), index=False, mode='w') # make a new file if not

def splitRecord():
	files = []
	files2= []

	truth = pd.read_csv("ground.csv", error_bad_lines=False) # label file csv
	total_rows = len(truth['bookingID'])
	for i in range(total_rows):
		if (truth['label'][i] == 0):
			files.append(str(truth['bookingID'][i]) + ".csv")
		else:
			files2.append(str(truth['bookingID'][i]) + ".csv")

	for f in files:
		if os.path.isfile(f):
			shutil.move(f, r'C:\Users\LEE\Desktop\python\safety\detect\test\0')

	for f in files2:
		if os.path.isfile(f):
			shutil.move(f, r'C:\Users\LEE\Desktop\python\safety\detect\test\1')

def getGroundLabel(label):
	#dirname = os.path.dirname(__file__)
	empty=[]
	labelList = []

	for i in label:
		labelList.append(os.path.basename(i))

	truth = pd.read_csv("ground.csv", error_bad_lines=False)
	total_rows = len(truth['bookingID'])
	for i in range(total_rows):
		filename = str(truth['bookingID'][i])+ ".csv"
		if filename not in labelList:
			empty.append(i)
	truth = truth.drop(truth.index[empty])
	truth.to_csv("ground2.csv",index=False)

def combineFiles():
	PATH = r'C:\Users\LEE\Desktop\python\safety\detect\test\\'
	EXT = "*.csv"
	all_csv_files = [file
	                 for path,subdir,files in os.walk(PATH)
	                 for file in glob(os.path.join(path, EXT))]#
	dangerous = []
	safe = []
	testSet = []
	header_saved = False

	for i in all_csv_files:
		path=os.path.dirname(i)
		folderName=os.path.basename(path)
		if (folderName == "1"):
			dangerous.append(i)
		elif(folderName == "0"): 
			safe.append(i)

	testSet = dangerous[:50] + safe[:50] # first ten from

	getGroundLabel(testSet)

	with open('combineRecord.csv','a') as fout:
		for tripRecord in testSet:
			with open(tripRecord) as fin:
				header = next(fin)
				if not header_saved:
					fout.write(header)
					header_saved = True
				for line in fin:
					fout.write(line)
		fout.close()

def preprocess():
	empty=[]
	df = pd.read_csv("combineRecord.csv", error_bad_lines=False, low_memory=False)
	
	total_rows = len(df['bookingID'])

	df = df.assign(totalAcceleration=0.0)
	df = df.assign(totalGyro=0.0)

	total_rows = len(df['bookingID'])
	for i in range(total_rows):

		X = df["acceleration_x"][i]**(2) + df["acceleration_y"][i]**(2) + df["acceleration_z"][i]**(2)
		df.at[i, 'totalAcceleration']= math.sqrt(X)

		Y = float(df["gyro_x"][i])**(2) + float(df["gyro_y"][i])**(2) + float(df["gyro_z"][i])**(2)
		df.at[i, 'totalGyro']= math.sqrt(Y)

	del df['Accuracy']
	del df['acceleration_x']
	del df['acceleration_y']
	del df['acceleration_z']
	del df['gyro_x']
	del df['gyro_y']
	del df['gyro_z']

	df = df.groupby('bookingID')
	df = df.apply(lambda _df: _df.sort_values(by=['second']))

	df.to_csv("combine2.csv",index=False)

def features():
	truth = pd.read_csv("ground2.csv", error_bad_lines=False)
	df = pd.read_csv("combine2.csv", error_bad_lines=False)

	kind_to_fc_parameters = {
    "Speed": {"maximum": None, "mean_abs_change": None,
    "count_above_mean": None, "longest_strike_above_mean": None},

    "totalAcceleration": {"maximum": None, "mean_abs_change": None,
    "count_above_mean": None, "longest_strike_above_mean": None},

    "totalGyro": {"maximum": None, "mean_abs_change": None,
    "count_above_mean": None, "longest_strike_above_mean": None},

    "Bearing":{"mean_abs_change": None,
    "count_above_mean": None, "longest_strike_above_mean": None}
	}

	features_filtered_direct = extract_relevant_features(df,tripLabel,
		column_id='bookingID', column_sort='second', kind_to_fc_parameters=kind_to_fc_parameters)
	print(features_filtered_direct.head())
	features_filtered_direct.to_csv("feature1000.csv",index=False)

def train():
	X_filtered = pd.read_csv("feature1000.csv", error_bad_lines=False)
	y = pd.read_csv("ground2.csv", error_bad_lines=False)
	tripLabel = pd.Series(data=y["label"].values,index = y["bookingID"].values)
	X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X_filtered, tripLabel, test_size=.2)

	cl2 = RandomForestClassifier(n_estimators=1000)#DecisionTreeClassifier()
	cl2.fit(X_filtered_train, y_train)
	probs = cl2.predict_proba(X_filtered_test) 
	probs = probs[:, 1]
	auc = roc_auc_score(y_test, probs)  
	print('AUC: %.2f' % auc)
	fpr, tpr, thresholds = roc_curve(y_test, probs)  
	plot_roc_curve(fpr, tpr) 

def plot_roc_curve(fpr, tpr):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def main():
	rawToRecord()
	#splitRecord()
	#combineFiles()
	#preprocess()
	#features()
	#train()


if __name__ == "__main__":
    main()