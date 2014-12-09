import numpy
import numpy as np
import random
import pandas as pd

import os
import csv

class FeatureParser:
	def __init__(self, messageFile,lobFile,timeFrame,useCachedFiles):
		self.messageFile = messageFile
		self.lobFile = lobFile
		self.indDict = self.getIndDict()
		self.timeFrame = timeFrame
		self.useCachedFiles = useCachedFiles

		self.intervals = [1,2,5,10,25,50,100,250,500,1000,2500,5000,10000]

	def getIndDict(self):
		ret = dict()
		ret['time'] = 0
		ret['type'] = 1
		ret['ID'] = 2
		ret['orderSize'] = 3
		ret['orderPrice'] = 4
		ret['dir'] = 5

		ret['ap1'] = 6
		ret['as1'] = 7
		ret['bp1'] = 8
		ret['bs1'] = 9

		ret['ap2'] = 10
		ret['as2'] = 11
		ret['bp2'] = 12
		ret['bs2'] = 13

		ret['ap3'] = 14
		ret['as3'] = 15
		ret['bp3'] = 16
		ret['bs3'] = 17

		ret['ap4'] = 18
		ret['as4'] = 19
		ret['bp4'] = 20
		ret['bs4'] = 21

		ret['ap5'] = 22
		ret['as5'] = 23
		ret['bp5'] = 24
		ret['bs5'] = 25
		return ret

	def parseMat(self, string):
		strfinal = string.replace("\n",";")
		strfinal = strfinal[:-1]
		return numpy.matrix(strfinal)

	def joinRaw(self, messageRaw, lobRaw):
		finalList = []

		messageSplit = messageRaw.split("\n")
		lobSplit = lobRaw.split("\n")

		if len(messageSplit) != len(lobSplit):
			print len(messageSplit)
			print len(lobSplit)
			raise Exception("messageSplit and lobSplit length unequal")

		for i in range(len(messageSplit) - 1):
			finalList.append("%s,%s\n" % (messageSplit[i],lobSplit[i]))

		return "".join(finalList)

	def removeDuplicateTimes(self,matRaw):
		[a,b] = numpy.unique(numpy.array(matRaw[:,0]),return_index=True)
		return matRaw[b,:]

	def addSpreadAndMidPrice(self,matRaw):
		for i in range(1,2):
			priceSpread = matRaw[:,self.indDict["ap%d" % i]] - matRaw[:,self.indDict["bp%d" % i]]
			midPrice = (matRaw[:,self.indDict["ap%d" % i]] + matRaw[:,self.indDict["bp%d" % i]]) / 2.

			matRaw = numpy.hstack((matRaw,priceSpread))
			matRaw = numpy.hstack((matRaw,midPrice))
		return matRaw


	def addMaxMinPriceSpreadDifference(self,matRaw):
		tempAskMat = numpy.zeros((len(matRaw),1))
		tempBidMat = numpy.zeros((len(matRaw),1))

		for i in range(1,2):
			tempAskMat = numpy.hstack((tempAskMat,matRaw[:,self.indDict["ap%d" % i]]))
			tempBidMat = numpy.hstack((tempBidMat,matRaw[:,self.indDict["bp%d" % i]]))

		numpy.delete(tempAskMat,0,axis=1)
		numpy.delete(tempBidMat,0,axis=1)

		matRaw = numpy.hstack((matRaw,numpy.amax(tempAskMat,axis=1) - numpy.amin(tempAskMat,axis=1)))
		matRaw = numpy.hstack((matRaw,numpy.amax(tempBidMat,axis=1) - numpy.amin(tempBidMat,axis=1)))

		return matRaw

	def addMeanPriceVolume(self,matRaw):
		tempAskPriceMat = numpy.zeros((len(matRaw),1))
		tempBidPriceMat = numpy.zeros((len(matRaw),1))
		tempAskVolMat = numpy.zeros((len(matRaw),1))
		tempBidVolMat = numpy.zeros((len(matRaw),1))

		for i in range(1,2):
			tempAskPriceMat += matRaw[:,self.indDict["ap%d" % i]]
			tempBidPriceMat += matRaw[:,self.indDict["bp%d" % i]]
			tempAskVolMat += matRaw[:,self.indDict["as%d" % i]]
			tempBidVolMat += matRaw[:,self.indDict["bs%d" % i]]

		matRaw = numpy.hstack((matRaw,tempAskPriceMat / 5.))
		matRaw = numpy.hstack((matRaw,tempBidPriceMat / 5.))
		matRaw = numpy.hstack((matRaw,tempAskVolMat / 5.))
		matRaw = numpy.hstack((matRaw,tempBidVolMat / 5.))

		return matRaw

	def addAccumDifference(self,matRaw):
		tempPriceDiff = numpy.zeros((len(matRaw),1))
		tempVolDiff = numpy.zeros((len(matRaw),1))

		for i in range(1,2):
			tempPriceDiff += matRaw[:,self.indDict["ap%d" % i]] - matRaw[:,self.indDict["bp%d" % i]]
			tempVolDiff += matRaw[:,self.indDict["as%d" % i]] - matRaw[:,self.indDict["bs%d" % i]]

		matRaw = numpy.hstack((matRaw, tempPriceDiff))
		matRaw = numpy.hstack((matRaw, tempVolDiff))

		return matRaw


	def addDerivatives(self,matRaw):
		shiftMatRaw = numpy.vstack((matRaw[0,:],matRaw)) 
		shiftMatRaw = numpy.delete(shiftMatRaw, len(shiftMatRaw) - 1,axis = 0)
		shiftMatRaw[0,self.indDict['time']] -= 1 #prevent divide by zero for slope

		timeCol = matRaw[:,self.indDict['time']] - shiftMatRaw[:,self.indDict['time']]

		for i in range(1,2):
			matRaw = numpy.hstack((matRaw, (1.*matRaw[:,self.indDict["ap%d" % i]] - shiftMatRaw[:,self.indDict["ap%d" % i]]) / (timeCol)))
			matRaw = numpy.hstack((matRaw, (1.*matRaw[:,self.indDict["bp%d" % i]] - shiftMatRaw[:,self.indDict["bp%d" % i]]) / (timeCol)))
			matRaw = numpy.hstack((matRaw, (1.*matRaw[:,self.indDict["as%d" % i]] - shiftMatRaw[:,self.indDict["as%d" % i]]) / (timeCol)))
			matRaw = numpy.hstack((matRaw, (1.*matRaw[:,self.indDict["bs%d" % i]] - shiftMatRaw[:,self.indDict["bs%d" % i]]) / (timeCol)))

		return matRaw

	def addRollingDerivatives(self,matRaw):
		for i in range(1,2):
			for j in self.intervals:
				ap = matRaw[:,self.indDict["ap%d"%i]]
				bp = matRaw[:,self.indDict["bp%d"%i]]
				time = 	matRaw[:,self.indDict["time"]]

				roll_ap = pd.stats.moments.rolling_apply(ap,j,lambda x: x[-1] - x[0])
				roll_bp = pd.stats.moments.rolling_apply(bp,j,lambda x: x[-1] - x[0])
				roll_time = pd.stats.moments.rolling_apply(time,j,lambda x: x[-1] - x[0])

				roll_ap[numpy.isnan(roll_ap)] = 0
				roll_bp[numpy.isnan(roll_bp)] = 0				
				roll_time[numpy.isnan(roll_time)] = 1				

				temp1 = roll_ap*1. / roll_time
				temp2 = roll_bp*1. / roll_time

				temp1[numpy.isnan(temp1)] = 0
				temp2[numpy.isnan(temp2)] = 0


				matRaw = numpy.hstack((matRaw, temp1))
				matRaw = numpy.hstack((matRaw, temp2))
		return matRaw
	#def addRollingSecondDerivatives(self,matRaw):

	#has NaN's
	def addRollingVariances(self,matRaw):
		for i in range(1,2):
			for j in self.intervals:
				ap = matRaw[:,self.indDict["ap%d"%i]]
				bp = matRaw[:,self.indDict["bp%d"%i]]

				roll_ap = pd.stats.moments.rolling_var(ap,j)
				roll_bp = pd.stats.moments.rolling_var(bp,j)

				#roll_ap[:j] = roll_ap[j]*numpy.ones((j,1))
				#roll_bp[:j] = roll_bp[j]*numpy.ones((j,1))

				roll_ap[numpy.isnan(roll_ap)] = -1
				roll_bp[numpy.isnan(roll_bp)] = -1

				matRaw = numpy.hstack((matRaw,roll_ap))
				matRaw = numpy.hstack((matRaw,roll_bp))

		return matRaw

	def addRollingMedian(self,matRaw):
		for i in range(1,2):
			for j in self.intervals:
				ap = matRaw[:,self.indDict["ap%d"%i]]
				bp = matRaw[:,self.indDict["bp%d"%i]]

				roll_ap = pd.stats.moments.rolling_median(ap,j)
				roll_bp = pd.stats.moments.rolling_median(bp,j)

				roll_ap[:j] = ap[:j]
				roll_bp[:j] = bp[:j]

				matRaw = numpy.hstack((matRaw,roll_ap))
				matRaw = numpy.hstack((matRaw,roll_bp))


		return matRaw

	def addRollingAverages(self,matRaw):
		#pd.stats.moments.rolling_mean(np.arange(12),6)
		for i in range(1,2):
			for j in self.intervals:
				ap = matRaw[:,self.indDict["ap%d"%i]]
				bp = matRaw[:,self.indDict["bp%d"%i]]

				roll_ap = pd.stats.moments.rolling_mean(ap,j)
				roll_bp = pd.stats.moments.rolling_mean(bp,j)

				roll_ap[:j] = ap[:j]
				roll_bp[:j] = bp[:j]

				matRaw = numpy.hstack((matRaw,roll_ap))
				matRaw = numpy.hstack((matRaw,roll_bp))


		return matRaw


	def addRollingMin(self,matRaw):
		for i in range(1,2):
			for j in self.intervals:
				ap = matRaw[:,self.indDict["ap%d"%i]]
				bp = matRaw[:,self.indDict["bp%d"%i]]

				roll_ap = pd.stats.moments.rolling_min(ap,j)
				roll_bp = pd.stats.moments.rolling_min(bp,j)

				roll_ap[:j] = roll_ap[j]*numpy.ones((j,1))
				roll_bp[:j] = roll_bp[j]*numpy.ones((j,1))

				matRaw = numpy.hstack((matRaw,roll_ap))
				matRaw = numpy.hstack((matRaw,roll_bp))


		return matRaw

	def addRollingMax(self,matRaw):
		for i in range(1,2):
			for j in self.intervals:
				ap = matRaw[:,self.indDict["ap%d"%i]]
				bp = matRaw[:,self.indDict["bp%d"%i]]

				roll_ap = pd.stats.moments.rolling_max(ap,j)
				roll_bp = pd.stats.moments.rolling_max(bp,j)

				roll_ap[:j] = roll_ap[j]*numpy.ones((j,1))
				roll_bp[:j] = roll_bp[j]*numpy.ones((j,1))

				matRaw = numpy.hstack((matRaw,roll_ap))
				matRaw = numpy.hstack((matRaw,roll_bp))


		return matRaw





	"""Expensive Label"""
	"""
	def label(self,matRaw):
		#Policy:
		#	If there is a ___ sometime before the end of (self.timeFrame) events, then label as ____
		#		1) Positive Spread Cross -> +1
		#		2) No Spread Cross -> 0
		#		3) Negative Spread Cross -> -1

		labelMat = numpy.zeros((len(matRaw),1))

		ap1 = matRaw[:,self.indDict['ap1']] 
		bp1 = matRaw[:,self.indDict['bp1']]

		timeFrame = self.timeFrame


		for currInd in range(len(matRaw) - timeFrame):
			if currInd % 1000 == 0:
				print currInd

			currLabelUp = False
			currLabelNeutral = False
			currLabelDown = False
			for i in range(currInd+1, currInd + timeFrame):
				if ap1[currInd] < bp1[i]:
					currLabelUp = True
				elif bp1[currInd] > ap1[i]:
					currLabelDown = True
				else:
					currLabelNeutral = True

			if currLabelUp == True and currLabelDown == False:
				labelMat[currInd] = 1
			elif currLabelUp == False and currLabelDown == True:
				labelMat[currInd] = -1
			else:
				labelMat[currInd] = 0
		#print labelMat
		print "Num positive cross: %d" % len(numpy.extract(labelMat == 1, labelMat))
		print "Num no cross: %d" % len(numpy.extract(labelMat == 0, labelMat))
		print "Num negative cross: %d" % len(numpy.extract(labelMat == -1, labelMat))

		return labelMat
	"""

	"""	
	def label(self,matRaw):
		#Policy:
		#	If there is a ___ sometime before the end of (self.timeFrame) events, then label as ____
		#		1) Positive Spread Cross -> +1
		#		2) No Spread Cross -> 0
		#		3) Negative Spread Cross -> -1

		labelMat = numpy.zeros((len(matRaw),1))

		ap1 = matRaw[:,self.indDict['ap1']] 
		bp1 = matRaw[:,self.indDict['bp1']]

		timeFrame = self.timeFrame

		for currInd in range(len(matRaw) - timeFrame):

			upCount = sum(ap1[currInd] < bp1[currInd+1:currInd+timeFrame])
			downCount = sum(bp1[currInd] > ap1[currInd+1:currInd+timeFrame])

			if currInd % 100 == 0:
				print upCount
				print downCount
				print currInd

			upProp,downProp = upCount*1./timeFrame, downCount*1./timeFrame
			if (upProp > downProp) and upProp > .5:
				labelMat[currInd] = 1
			elif (upProp < downProp) and downProp > .5:
				labelMat[currInd] = -1
			else:
				labelMat[currInd] = 0


		#print labelMat
		print "Num positive cross: %d" % len(numpy.extract(labelMat == 1, labelMat))
		print "Num no cross: %d" % len(numpy.extract(labelMat == 0, labelMat))
		print "Num negative cross: %d" % len(numpy.extract(labelMat == -1, labelMat))

		return labelMat
	"""
	def pseudoRotate(self,l,n):
		curr = l[:,0].transpose().tolist()[0]
		curr = curr[n:] + [curr[-1]]*n
		return curr

	def label(self,matRaw):
		labelMat = numpy.zeros((len(matRaw),1))

		ap1 = matRaw[:,self.indDict['ap1']] 
		bp1 = matRaw[:,self.indDict['bp1']]

		timeFrame = self.timeFrame
		ap1Mat = []
		bp1Mat = []

		for i in range(1,timeFrame+1):
			if i % 100 == 0:
				print "curr rotation %d" % i
			ap1Mat.append(self.pseudoRotate(ap1,i))
			bp1Mat.append(self.pseudoRotate(bp1,i))

		ap1Mat = numpy.mat(ap1Mat).transpose()
		print "Converted ap1Mat"
		bp1Mat = numpy.mat(bp1Mat).transpose()
		print "Converted bp1Mat"

		upBoolCol = numpy.array(numpy.sum(ap1 < bp1Mat,axis=1) *1. / timeFrame)
		downBoolCol = numpy.array(numpy.sum(bp1 > ap1Mat,axis=1)*1. / timeFrame)

		a_boolCol = (upBoolCol > downBoolCol) * (upBoolCol > .75)
		b_boolCol = (upBoolCol < downBoolCol) * (downBoolCol > .75) * -1

		return numpy.mat(a_boolCol + b_boolCol)

	def parse(self):
		
		if os.path.isfile("xtrain_time15.csv") and os.path.isfile("ytrain_time15.csv") and self.useCachedFiles == True:
			print "Using cached files"
			xtrain_reader=csv.reader(open("xtrain_time15.csv","rb"),delimiter=',')
			x_train=list(xtrain_reader)
			x_train=numpy.mat(x_train).astype('float')

			ytrain_reader=csv.reader(open("ytrain_time15.csv","rb"),delimiter=',')
			y_train=list(ytrain_reader)
			y_train=numpy.mat(y_train).astype('float')

			print x_train.shape
			print y_train.shape
			return x_train,y_train			


		messageRaw = file(self.messageFile).read()
		lobRaw = file(self.lobFile).read()

		print "1"
		concatRaw = self.joinRaw(messageRaw,lobRaw)
		print "2"
		matRaw = self.parseMat(concatRaw)
		print "3"
		matRaw = self.removeDuplicateTimes(matRaw)
		print "4"
		matRaw = self.addSpreadAndMidPrice(matRaw)
		print "5"
		matRaw = self.addMaxMinPriceSpreadDifference(matRaw)
		print "6"
		matRaw = self.addMeanPriceVolume(matRaw)
		print "7"
		matRaw = self.addAccumDifference(matRaw)
		print "8"
		matRaw = self.addDerivatives(matRaw)
		print "9"
		matRaw = self.addRollingMin(matRaw)
		print "10"
		matRaw = self.addRollingMax(matRaw)
		print "11"
		matRaw = self.addRollingAverages(matRaw)
		print "12"
		matRaw = self.addRollingVariances(matRaw)
		print "13"
		matRaw = self.addRollingMedian(matRaw)
		print "14"
		matRaw = self.addRollingDerivatives(matRaw)

		print "finalMat has nan %s" % numpy.isnan(matRaw).any()

		self.finalMat = matRaw
		numpy.savetxt("xtrain_time15.csv", self.finalMat, delimiter=",")

		print "saved xtrain"

		self.labelMat = self.label(matRaw)
		print "labelMat has nan %s" % numpy.isnan(self.labelMat).any()

		numpy.savetxt("ytrain_time15.csv", self.labelMat, delimiter=",")
		print "saved ytrain"

		print self.finalMat.shape
		print self.labelMat.shape

		return self.finalMat, self.labelMat

#messageBook = "GOOG_2012-06-21_34200000_57600000_message_5.csv"
#limitBook = "GOOG_2012-06-21_34200000_57600000_orderbook_5.csv"

#messageBook = "a.csv"
#limitBook = "b.csv"
#q = FeatureParser(messageBook,limitBook,200)
#q.parse()
