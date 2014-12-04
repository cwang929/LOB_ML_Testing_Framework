import numpy
import random

class FeatureParser:
	def __init__(self, messageFile, lobFile,timeFrame):
		self.messageFile = messageFile
		self.lobFile = lobFile
		self.indDict = self.getIndDict()
		self.timeFrame = timeFrame

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
		#print matRaw[:,0]
		#print matRaw.shape
		#print type(matRaw[0,0])
		[a,b] = numpy.unique(numpy.array(matRaw[:,0]),return_index=True)
		return matRaw[b,:]

	def addSpreadAndMidPrice(self,matRaw):
		for i in range(1,6):
			priceSpread = matRaw[:,self.indDict["ap%d" % i]] - matRaw[:,self.indDict["bp%d" % i]]
			midPrice = (matRaw[:,self.indDict["ap%d" % i]] + matRaw[:,self.indDict["bp%d" % i]]) / 2.

			matRaw = numpy.hstack((matRaw,priceSpread))
			matRaw = numpy.hstack((matRaw,midPrice))
		return matRaw


	def addMaxMinPriceDifference(self,matRaw):
		tempAskMat = numpy.zeros((len(matRaw),1))
		tempBidMat = numpy.zeros((len(matRaw),1))

		for i in range(1,6):
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

		for i in range(1,6):
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

		for i in range(1,6):
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

		for i in range(1,6):
			matRaw = numpy.hstack((matRaw, (1.*matRaw[:,self.indDict["ap%d" % i]] - shiftMatRaw[:,self.indDict["ap%d" % i]]) / (timeCol)))
			matRaw = numpy.hstack((matRaw, (1.*matRaw[:,self.indDict["bp%d" % i]] - shiftMatRaw[:,self.indDict["bp%d" % i]]) / (timeCol)))
			matRaw = numpy.hstack((matRaw, (1.*matRaw[:,self.indDict["as%d" % i]] - shiftMatRaw[:,self.indDict["as%d" % i]]) / (timeCol)))
			matRaw = numpy.hstack((matRaw, (1.*matRaw[:,self.indDict["bs%d" % i]] - shiftMatRaw[:,self.indDict["bs%d" % i]]) / (timeCol)))

		return matRaw

	"""	
	def label(self,matRaw):
		#Policy:
		#	If there is a ___ exactly after 5 events, then label as ____
		#		1) Positive Spread Cross -> +1
		#		2) No Spread Cross -> 0
		#		3) Negative Spread Cross -> -1

		labelMat = numpy.zeros((len(matRaw),1))

		ap1 = matRaw[:,self.indDict['ap1']] 
		bp1 = matRaw[:,self.indDict['bp1']]

		timeFrame = self.timeFrame

		for currInd in range(len(matRaw) - timeFrame):
			futureInd = currInd + timeFrame
			if ap1[currInd] < bp1[futureInd]:
				labelMat[currInd] = 1
			elif bp1[currInd] > ap1[futureInd]:
				labelMat[currInd] = -1
			else:
				labelMat[currInd] = 0
		#print labelMat
		print "Num positive cross: %d" % len(numpy.extract(labelMat == 1, labelMat))
		print "Num no cross: %d" % len(numpy.extract(labelMat == 0, labelMat))
		print "Num negative cross: %d" % len(numpy.extract(labelMat == -1, labelMat))

		return labelMat
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
	

	def parse(self):
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
		matRaw = self.addMaxMinPriceDifference(matRaw)
		print "6"
		matRaw = self.addMeanPriceVolume(matRaw)
		print "7"
		matRaw = self.addAccumDifference(matRaw)
		print "8"
		matRaw = self.addDerivatives(matRaw)
		print "9"

		self.finalMat = matRaw
		numpy.savetxt("xtrain_time15.csv", self.finalMat, delimiter=",")

		self.labelMat = self.label(matRaw)
		numpy.savetxt("ytrain_time15.csv", self.labelMat, delimiter=",")

		return self.finalMat, self.labelMat

#messageBook = "GOOG_2012-06-21_34200000_57600000_messagebook_5.csv"
#limitBook = "GOOG_2012-06-21_34200000_57600000_orderbook_5.csv"

#messageBook = "a.csv"
#limitBook = "b.csv"
#q = FeatureParser(messageBook,limitBook,200)
#q.parse()
