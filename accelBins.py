import numpy as np


class accelBins:
        def __init__(self):
                self.bins = []
                self.max = 50.0000
                self.min = -50.0000
                self.step = 0
                self.len = 0


        def initialize(self, accelerations):
                self.bins = np.linspace(self.min, self.max, 100, endpoint=True)
                self.bins.fill(0)
                self.step = (self.max - self.min) / (len(self.bins) - 1)
                self.len = len(accelerations)
                for accel in accelerations:
                        index = round((accel - self.min) / self.step)
                        index = max(0, min(index, len(self.bins) - 1))
                        self.bins[index] = self.bins[index] + 1    



                            

        def getFreq(self, accel):

                if ((accel > self.max) or (accel < self.min)):
                        return 0
                
                else:
                        index = round((accel - self.min) / self.step)
                        index = max(0, min(index, len(self.bins) - 1))
                        counts = self.bins[index]
                        return counts / self.len
                

        def getLen(self):
                return self.len
