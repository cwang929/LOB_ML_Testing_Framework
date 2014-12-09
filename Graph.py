import matplotlib.pyplot as plt
import numpy as np
import numpy

class Graph:
	def __init__(self, messageBook, limitBook):
		self.messageBook = messageBook
		self.limitBook = limitBook

	def plot(self):
		messageFile = open(self.messageBook)
		limitFile = open(self.limitBook)

		#totalMat = None

		ap1 = []
		as1 = []
		bp1 = []
		bs1 = []

		count = 0

		for (messageLine, limitLine) in map(None, messageFile,limitFile):
			if messageLine == None or limitLine == None:
				break

			rawLine = ",".join([messageLine,limitLine])
			rawLine = rawLine.replace("\n","")
			feature = numpy.mat(rawLine).tolist()[0]

			ap1.append(feature[6] / 10000.)
			as1.append(feature[7])
			bp1.append(feature[8] / 10000. )
			bs1.append(feature[9])

			"""
			if totalMat == None:
				totalMat = feature
				print "here"
			else:
				totalMat = numpy.vstack((totalMat,feature))
			"""

			if count % 1000 == 0:
				print count

			count += 1

		#print totalMat.shape


		#ap1 = (totalMat[:,6].transpose()/10000.).tolist()[0]
		#bp1 = (totalMat[:,8].transpose()/10000.).tolist()[0]

		print len(ap1)
		print len(bp1)

		fig = plt.subplots()
		ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
		ax2 = plt.subplot2grid((3,1), (2,0))
		x = range(len(ap1))

		ax1.plot(x,ap1,'r')
		ax1.plot(x,bp1,'b')
		ax1.set_xlabel('# of stock events')
		ax1.set_ylabel('price')

		#ax2 = ax1.twinx()
		ax2.plot(x, as1, 'g')
		ax2.plot(x, bs1, 'c')
		ax2.set_ylabel('volume')

		plt.show()







messageBook = "GOOG_2012-06-21_34200000_57600000_message_5.csv"
limitBook = "GOOG_2012-06-21_34200000_57600000_orderbook_5.csv"

#messageBook = "GOOG_messagebook_5_small.csv"
#limitBook = "GOOG_orderbook_5_small.csv"

q = Graph(messageBook,limitBook)
q.plot()