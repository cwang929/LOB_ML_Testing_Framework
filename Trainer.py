import numpy
import scipy
import Model

from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

class Trainer:
	def __init__(self,modelType):
		self.modelType = modelType

	def getFinalMatAndLabels(self,yesMat, noMat):
		yesLabel = numpy.ones((len(yesMat),1))
		noLabel = numpy.zeros((len(noMat),1))

		return numpy.vstack((yesMat,noMat)), numpy.vstack((yesLabel,noLabel))

	def getModel(self):
		if self.modelType == "DecisionTree":
			return DecisionTreeClassifier(max_depth=35, min_samples_split=1,random_state=0)#DecisionTreeClassifier(max_depth=35, min_samples_split=1,random_state=0)
		elif self.modelType == "RandomForest":
			return RandomForestClassifier(n_estimators=10, max_depth=35, min_samples_split=1, random_state=0)
		elif self.modelType == "SVM":
			return SVC(kernel='poly') #rbf
			#Default: SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
			#		 gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
			#		 shrinking=True, tol=0.001, verbose=False)
			
		elif self.modelType == "ExtraTrees":
			return ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
		elif self.modelType == "AdaBoost":
			return AdaBoostClassifier(n_estimators=100)
		else:
			raise Exception("%s has not been implemented yet" % self.modelType)
		return None

	def train(self,dataList):
		neutralMat,upMat,downMat = dataList


		print "here"
		neutralTrain,neutralLabels = self.getFinalMatAndLabels(neutralMat, numpy.vstack((upMat,downMat)))
		neutralModel = self.getModel().fit(neutralTrain,neutralLabels)

		upTrain,upLabels = self.getFinalMatAndLabels(upMat, numpy.vstack((neutralMat,downMat)))
		upModel = self.getModel().fit(upTrain,upLabels)

		downTrain,downLabels = self.getFinalMatAndLabels(downMat, numpy.vstack((neutralMat,upMat)))
		downModel = self.getModel().fit(downTrain,downLabels)

		return Model.Model(neutralModel,upModel,downModel)
	