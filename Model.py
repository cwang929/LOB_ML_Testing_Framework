class Model:
	def __init__(self,neutralModel,upModel,downModel):
		self.neutralModel = neutralModel
		self.upModel = upModel
		self.downModel = downModel

	def predict(self,obs):
		if (self.neutralModel == None
		or self.upModel == None
		or self.downModel == None):
			return 0

		n = self.neutralModel.predict(obs)[0]
		u = self.upModel.predict(obs)[0]
		d = self.downModel.predict(obs)[0]

		if (n,u,d) == (0,1,0) or (n,u,d) == (1,1,0):
			return 1
		elif (n,u,d) == (0,0,1) or (n,u,d) == (0,1,1):
			return -1
		else:
			return 0
	