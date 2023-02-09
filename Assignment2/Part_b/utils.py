class Dataset:
    def __init__(self,no_of_points):
        
        self.n=no_of_points
    def get(self,add_noise=False):
        import pandas as pd
        import numpy as np
        X=[]
        Y=[]
        Labels=[]
        K=[0,3]
        sign=[-1,1]
        for i in range(self.n):
            
            k=np.random.choice(K)
            x=np.random.uniform(-1,1)
            X.append(x)
            y = np.power(1-(x**2),0.5)
            y=np.random.choice(sign)*y
            y=y+k
            Y.append(y)
            if(k==0):
                Labels.append(0)
            else:
                Labels.append(1)

        if(add_noise==True):
            for i in range(self.n):
                X[i]+=np.random.normal(0, 0.1)
                Y[i]+=np.random.normal(0, 0.1)
                
                
        data={'X':X,'Y':Y,'Labels':Labels}
        df=pd.DataFrame(data)
        
        return df

class perceptron():
    def predict(self,row,):
        activation = self.weights[0]
        for i in range(len(row)-1):
            activation += self.weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0

    # Estimate Perceptron weights using stochastic gradient descent
    def train_weights(self,train,n_epoch,bias):
        self.weights = [1.0 for i in range(len(train[0]))]
        self.weights[0]=0.0
        for epoch in range(n_epoch):
            sum_error = 0.0
            for row in train:
                prediction = self.predict(row)
                error = row[-1] - prediction
                sum_error += error**2
                if(bias==1):
                    self.weights[0] = self.weights[0] +  error 
                else:
                    self.weights[0]=0.0
                for i in range(len(row)-1):
                    self.weights[i + 1] = self.weights[i + 1] + error * row[i]
            print('>epoch=%d, error=%.3f' % (epoch, sum_error),end=' ')
            print(self.weights)

    def make_prediction(self,data):
        import numpy as np
        return np.dot(self.weights[1:],data)+self.weights[0]

def train_test_split(data,t):
    x=len(data)
    print(x)
    x=int(x*t//100)
    print(x)
    return data[:x,:-1] , data[:x,-1] ,data[x:,:-1] ,data[x:,-1]



