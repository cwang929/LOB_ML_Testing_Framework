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
		else:
			self.list.append(val)
		return retVal

	def isSaturated(self):
		return len(self.list) == self.maxLen


class Cache:
	def __init__(self,numNeutral,numUp,numDown,lookAheadTime):
		self.numNeutral = numNeutral
		self.numDown = numDown
		self.numUp = numUp
		self.lookAheadTime = lookAheadTime

		self.neutralQueue = Queue(numNeutral)
		self.upQueue = Queue(numUp)
		self.downQueue = Queue(numDown)

		self.tempQueue = Queue(self.lookAheadTime)

		self.counter = 0

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
			self.upQueue.push(outputVal)
		elif outputVal[0,bp1] > inputVal[0,ap1]:
			#label = -1
			self.downQueue.push(outputVal)
		else:
			#label = 0
			self.neutralQueue.push(outputVal)

		self.counter += 1

	def needUpdate(self):
		if ((self.counter < self.numNeutral + self.numDown + self.numUp)
		or self.neutralQueue.isSaturated() == False
		or self.upQueue.isSaturated() == False
		or self.downQueue.isSaturated() == False):
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

		retMat = queue.list.pop(0)

		while len(queue.list) > 0:
			tempRow = queue.list.pop(0)
			retMat = numpy.vstack((retMat, tempRow))

		return retMat



