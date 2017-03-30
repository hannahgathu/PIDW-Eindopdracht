import numpy as np
import csv
from NeuralNetwork import NeuralNetwork

m = 150
label = {"Iris-setosa" : 1, "Iris-versicolor" : 2, "Iris-virginica" : 3}
X = np.zeros((m,4))
Y = np.zeros(m)

with open('Iris.csv', newline='') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     #skip headers
     next(reader, None)
   
     #loop over rows
     for row in reader:
         #index
         k = int(row[0])-1
         # features
         X[k,:] = row[1:5]
         #labels
         Y[k] = label[row[5]]

# Now, X contains 150 input samples with 4 features each:
#(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
# and y contains 150 labels (1, 2, 3)
# Dit is de trainingsdata

X = X/10
Y = Y/3

n = NeuralNetwork(X, Y)
n.train(1000, 0.1)
print(3*n.predict([6.3,2.5,4.9,1.5]))
print('Weights               : ', n.p)
print('Output After Training : ',
      np.round(3/(1+np.exp(-(np.dot(X,n.p[:-1]) + n.p[-1])))))

      
