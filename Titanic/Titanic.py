from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np
import scipy as sp
class Titanic:
	def genfromtxtReader (self,trainFilePath,testFilePath):
		X = np.genfromtxt(trainFilePath, delimiter=';', usecols= (2,5,6,7,9))
		print(X)
	def csvReader(self,filePath):
		lines= csv.reader(open(filePath,"r+"))	
		dataset = list(lines)		
		return dataset
	def modelCreation (self,trainFilePath,testFilePath):
		dataSet = self.csvReader(trainFilePath)		
		dataSet = np.array(dataSet)	
		dataSet= sp.delete(dataSet,0,0)
		X = dataSet[:,[2,5,6,7,9]]
		print(type(X))		
		X.astype(np.float)		
		print(X)		
		#y = dataSet[:,[1]]		
		#clf = KNeighborsClassifier(n_neighbors = 5)
		#clf.fit(X,y)
		#return clf


if __name__ == "__main__":
	genFilePath = "/home/vishal/ML/Kaggle_Competition/Titanic/gender_submission.csv"
	testFilePath = "/home/vishal/ML/Kaggle_Competition/Titanic/test.csv"
	trainFilePath = "/home/vishal/ML/Kaggle_Competition/Titanic/train.csv"
	obj = Titanic()
	obj.modelCreation(trainFilePath,testFilePath)
	

	
	
