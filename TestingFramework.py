import numpy
import scipy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

import FeatureGenerator
import Model
import Cache
import Trainer
import Trader

class TestingFramework:
	def __init__(self,messageBook,limitBook,modelType):
		self.messageBook = messageBook
		self.limitBook = limitBook
		self.dataMat = None
		self.labelMat = None #note this is ground truth, will not be known to the trainer/model until X timesteps ahead
		self.predictMat = None

		self.timeFrame = 100

		"""
		self.timeFrame = 100 with the funky gaussian kernel
		______________
		Num Errors: 6294
		Error Rate: 0.344631
		# 1's predicted: 3737
		# 0's predicted: 12674
		# -1's predicted: 1852
		# 1's predicted but were 1's: 215
		# 1's predicted but were 0's: 3469
		# 1's predicted but were -1's: 53
		# 1 Success Rate: 0.057533
		# 0's predicted but were 1's: 321
		# 0's predicted but were 0's: 11745
		# 0's predicted but were -1's: 608
		# -1 Success Rate: 0.926700
		# -1's predicted but were 1's: 70
		# -1's predicted but were 0's: 1773
		# -1's predicted but were -1's: 9
		# -1 Success Rate: 0.004860
		_______________
		"""
		
		self.neutralQueueSize = 1000
		self.upQueueSize =  200
		self.downQueueSize = 200
		self.price = 1000000

		self.cache = Cache.Cache(self.neutralQueueSize,self.upQueueSize,self.downQueueSize,self.timeFrame)
		self.model = Model.Model(None, None, None)
		self.modelType = modelType
		self.trainer = Trainer.Trainer(self.modelType)
		self.trader = Trader.Trader(self.price,self.timeFrame)

	def errorRate(self,ans, actual):
		if (len(ans) != len(actual)):
			raise Exception("length of ans and actual vectors are not the same")

		errors = 0
		for i in range(len(ans)):
			if ans[i,0] != actual[i,0]:
				errors += 1
		print "Num Errors: %d" % errors
		return errors * 1. / len(actual)


	def numCountRow(self,mat,row):
		count = 0
		for elem in mat:
			elem = numpy.mat(elem)
			if elem[0,0] == row[0,0] and elem[0,1] == row[0,1]:
				count += 1
		return count



	def main(self):
		[dataMat, labelMat] = FeatureGenerator.FeatureParser(self.messageBook,self.limitBook,self.timeFrame).parse()
		print "Generated Features of size: (%d,%d)" % dataMat.shape
		pairwise_dists = squareform(pdist(dataMat, 'euclidean'))
		print "huh"
		self.dataMat = scipy.exp(-pairwise_dists**2 / 1**2)#dataMat
		print "whut"
		self.labelMat = labelMat

		countDict = dict()
		countDict[0] = 0
		countDict[1] = 0
		countDict[-1] = 0

		count = 0

		self.predictMat = numpy.zeros((labelMat.shape))

		for obs in dataMat: #an observation is a row in the dataMat
			self.cache.add(obs)
			if self.cache.needUpdate():
				#print "huh"
				self.model = self.trainer.train(self.cache.getData())
			prediction = self.model.predict(obs)
			self.predictMat[count,0] = prediction
			countDict[prediction] += 1
			count += 1
			#if count % 100 == 0:
				#print count, countDict
			self.trader.trade(Trader.DataPt(obs),prediction)

			#self.database.update(decision)

		lossMat = numpy.mat([[-1, 1, 2], [0, 0, 0], [2, 1, -1]])
		print "\n\nBegin Analysis\n______________"

		print "Model Metadata"
		print "Model Type: %s" % self.modelType
		print "Time Frame: %s" % self.timeFrame
		print "QueueSize (N,U,D): (%d,%d,%d)" % (self.neutralQueueSize,self.upQueueSize,self.downQueueSize)


		print "Error Rates\n_____________"
		print "Error Rate: %f" % self.errorRate(self.predictMat, self.labelMat)

		a = len(numpy.extract(self.labelMat == 1, self.labelMat))
		b = len(numpy.extract(self.labelMat == 0, self.labelMat))
		c = len(numpy.extract(self.labelMat == -1, self.labelMat))
		p1 = 1.*a/(a + b + c)
		p0 = 1.*b/(a + b + c)
		pn1 = 1.*c/(a + b + c)
		p = numpy.mat([pn1, p0, p1])

		print "# 1's labelled: %d" % len(numpy.extract(self.labelMat == 1, self.labelMat))
		print "# 0's labelled: %d" % len(numpy.extract(self.labelMat == 0, self.labelMat))
		print "# -1's labelled: %d" % len(numpy.extract(self.labelMat == -1, self.labelMat))

		print "# 1's predicted: %d" % len(numpy.extract(self.predictMat == 1, self.predictMat))
		print "# 0's predicted: %d" % len(numpy.extract(self.predictMat == 0, self.predictMat))
		print "# -1's predicted: %d" % len(numpy.extract(self.predictMat == -1, self.predictMat))

		N = len(self.predictMat)
		temp = numpy.hstack((self.predictMat,self.labelMat))
		a = self.numCountRow(temp,numpy.mat('1,1'))
		b = self.numCountRow(temp,numpy.mat('1,0'))
		c = self.numCountRow(temp,numpy.mat('1,-1'))
		v3 = [1.*a/N, 1.*b/N, 1.*c/N]
		print "# 1's predicted but were 1's: %d" % a
		print "# 1's predicted but were 0's: %d" % b
		print "# 1's predicted but were -1's: %d" % c
		print "# 1's Success Rate: %f" % (1 if (a,b,c) == (0,0,0) else (1.*a / (a + b + c)))

		a = self.numCountRow(temp,numpy.mat('0,1'))
		b = self.numCountRow(temp,numpy.mat('0,0'))
		c = self.numCountRow(temp,numpy.mat('0,-1'))
		v2 = [1.*a/N, 1.*b/N, 1.*c/N]
		print "# 0's predicted but were 1's: %d" % self.numCountRow(temp,numpy.mat('0,1'))
		print "# 0's predicted but were 0's: %d" % self.numCountRow(temp,numpy.mat('0,0'))
		print "# 0's predicted but were -1's: %d" % self.numCountRow(temp,numpy.mat('0,-1'))
		print "# 0's Success Rate: %f" % (1 if (a,b,c) == (0,0,0) else (1.*b / (a + b + c)))

		a = self.numCountRow(temp,numpy.mat('-1,1'))
		b = self.numCountRow(temp,numpy.mat('-1,0'))
		c = self.numCountRow(temp,numpy.mat('-1,-1'))
		v1 = [1.*a/N, 1.*b/N, 1.*c/N]
		print "# -1's predicted but were 1's: %d" % a
		print "# -1's predicted but were 0's: %d" % b
		print "# -1's predicted but were -1's: %d" % c
		print "# -1's Success Rate: %f" % (1 if (a,b,c) == (0,0,0) else (1.*c / (a + b + c)))

		print "Actual loss: %f" % numpy.sum(numpy.multiply(lossMat, numpy.mat([v1, v2, v3])))
		print "Expected loss while guessing according to prior distribution: %f" % numpy.sum(p*lossMat)

		print "_______________\n\n"

		total_value, trade_events = self.trader.metaData()

		print "Final Value %f" % total_value[-1]

		plt.plot(total_value)
		#plt.axis([0, len(trade_events), total_value[-1], 1200000])
		plt.ylabel('Portfolio label')
		plt.xlabel('Trade events')
		plt.show()

		#curr = self.cache.getData()
		#print curr
		#print curr[0].shape
		#print curr[1]
		#print curr[2].shape

		#numpy.savetxt("finalportfoliodata.csv", self.database.outputData(),delimiter=",")


messageBook = "GOOG_2012-06-21_34200000_57600000_messagebook_5.csv"
limitBook = "GOOG_2012-06-21_34200000_57600000_orderbook_5.csv"

q = TestingFramework(messageBook,limitBook,"RandomForest")
q.main()
