import numpy as np
from MultiLayerNeuralNetwork import MultiLayerNeuralNetwork

# invoer voor OF operatie (traininginput)
X = np.array([  [0,0],
                [0,1],
                [1,0],
                [1,1]])
    
# uitvoer OF (trainingoutput)       
Y = np.array([0,1,1,1])

### invoer voor EN operatie (traininginput)
##X = np.array([  [0,0],
##                [0,1],
##                [1,0],
##                [1,1]])
##    
### uitvoer EN (trainingoutput)       
##Y = np.array([0,0,0,1])
    
netwerk = MultiLayerNeuralNetwork([],X, Y)
netwerk.train(10,1)

   

