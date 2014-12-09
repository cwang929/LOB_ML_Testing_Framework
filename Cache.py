import numpy

class Queue:
	def __init__(self,maxLen):
		self.maxLen = maxLen
		self.list = []

	def push(self,val):
		retVal = None
		if len(self.list) == self.maxLen:
			retVal = self.list.pop(0)
		self.list.append(val)
		return retVal

	def isSaturated(self):
		return len(self.list) == self.maxLen

class Cache:
	def __init__(self,numNeutral,numUp,numDown,lookAheadTime,retrainCount):
		self.numNeutral = numNeutral
		self.numDown = numDown
		self.numUp = numUp
		self.lookAheadTime = lookAheadTime

		self.neutralQueue = Queue(numNeutral)
		self.upQueue = Queue(numUp)
		self.downQueue = Queue(numDown)

		self.tempQueue = Queue(self.lookAheadTime)

		self.counter = 0
		self.retrainCount = retrainCount

	def add(self,inputVal):
		ap1 = 6
		bp1 = 8

		outputVal = self.tempQueue.push(inputVal)

		if outputVal == None:
			return

		label = None

		#print inputVal
		#print inputVal.shape
		if outputVal[0,ap1] < inputVal[0,bp1]:
			#label = 1
			self.upQueue.push(outputVal.tolist()[0])
		elif outputVal[0,bp1] > inputVal[0,ap1]:
			#label = -1
			self.downQueue.push(outputVal.tolist()[0])
		else:
			#label = 0
			self.neutralQueue.push(outputVal.tolist()[0])

		self.counter += 1

	def needUpdate(self):
		if ((self.counter < self.retrainCount)
		or self.neutralQueue.isSaturated() == False
		or self.upQueue.isSaturated() == False
		or self.downQueue.isSaturated() == False):
		#(self.counter < self.numNeutral + self.numDown + self.numUp):
			return False
		else:
			return True


	def getData(self):
		self.counter = 0
		return map(self.convertQueueToMat, [self.neutralQueue, self.upQueue,self.downQueue])

	def convertQueueToMat(self, queue):
		if len(queue.list) == 0:
			return None

		if queue.isSaturated() == False:
			raise Exception("Queue not saturated")

		retMat = queue.list
		#queue.list = []

		return numpy.mat(retMat)
