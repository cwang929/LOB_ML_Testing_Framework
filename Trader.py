import numpy as np

class DataPt:
  data = None
  def __init__(self, data):
    self.data = data
  def bestAskPrice(self):
    return self.data[0, 6]
  def bestBidPrice(self):
    return self.data[0, 8]

class Trader:
  predictor = None
  #Initial state
  IN, shares, bought, time, last_in_time, last_ask, last_bid = False, 0, False, 0, 0, 0, 0
  money = None
  trade_events = []
  total_value = None
  predicted_crossings = 0

  #strategy : Self -> DataPt -> Predicted Midpoint Crossing -> Action,
  #Action : {-1,0,1} Sell, Hold, Buy
  def __init__(self, init_money,timeFrame):
    self.money = [init_money]
    self.stock_value = [0]
    self.total_value = [init_money]
    self.timeFrame = timeFrame
  def trade(self, datapt, midPtCross):
    action = 0
    if not self.IN:
      if midPtCross != 0:
        print ()
        #Predicted midpoint cross uptick. Buying maximum possible shares
        if midPtCross == 1:
          self.bid(datapt)
          self.bought = True
          action = 1
        #Predicted midpoint cross downtick. Selling maximum possible shares.
        else:
          self.ask(datapt)
          self.bought = False
          action = -1
        self.IN = True
        self.last_ask = datapt.bestAskPrice()
        self.last_bid = datapt.bestBidPrice()
      self.last_in_time = self.time
    elif self.IN:
      if ((self.time - self.last_in_time == self.timeFrame) or (self.bought and datapt.bestBidPrice() > self.last_ask) or (not self.bought and self.last_bid > datapt.bestAskPrice())):
      #if ((self.time - self.last_in_time == self.timeFrame)
      #or (datapt.bestBidPrice() > self.last_ask+.5)
      #or (datapt.bestAskPrice() < self.last_bid-.5)):
        #
        #upward or downward loss prevention
      #if ((self.time - self.last_in_time == self.timeFrame) or (self.bought and datapt.bestBidPrice() > self.last_ask) or  (not self.bought and self.last_bid > datapt.bestAskPrice())):
        #We bought and we've observed a midprice uptick (or 100 timesteps passed). So we sell.
        if self.bought:
          self.ask(datapt)
          action = -1
        #We sold and we've observed a midprice downtick (or 100 timesteps passed). So we buy.
        else:
          self.bid(datapt)
          action = 1
        self.IN = False
    self.trade_events.append((action, datapt))
    self.total_value.append(self.money[-1] + self.stock_value[-1])
    self.time += 1

  def bid(self, datapt):
    self.shares += 1.*self.money[-1]/datapt.bestAskPrice()
    self.stock_value.append(datapt.bestAskPrice()*self.shares)
    self.money.append(0)
    print self.money[-1], self.stock_value[-1],self.money[-1] + self.stock_value[-1], "1"
  def ask(self, datapt):
    self.money.append(self.money[-1] + self.shares*datapt.bestBidPrice())
    self.stock_value.append(0)
    self.shares = 0
    print self.money[-1], self.stock_value[-1],self.money[-1] + self.stock_value[-1], "2"
  def metaData(self):
    action, datapt = self.trade_events[-1]
    return (self.total_value, self.trade_events)

test = False
if test:
  X = np.genfromtxt('xtrain_time15.csv', delimiter=',')
  P = np.random.randint(3, size=100) - 1
  trader = Trader(1000000.0)
  N, d = X.shape
  for i in range(N):
    trader.trade(DataPt(X[i]), P[i])
  total_value, trade_events = trader.metaData()
  print total_value












