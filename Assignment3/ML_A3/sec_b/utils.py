class Mnn():
    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax','leaky_relu']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers = 3, layer_sizes = [768,1,10], activation = "tanh", learning_rate = 0.01, weight_init = "zero", batch_size = 128, num_epochs = 100, convergence = None):
        self.min_loss = 100000000
        self.weights = []    
        self.biases = []
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.convergence = convergence

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')
        else:
            self.activation = activation
    
        self.learning_rate =learning_rate
        
        
        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')
        else:
            self.weight_init = weight_init
            
            
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        if(weight_init=="zero"):
            for i in range(self.n_layers-1):
                weight = self.zero_init(shape =(self.layer_sizes[i],self.layer_sizes[i+1]))
                self.weights.append(weight)
        elif(weight_init=="random"):
            for i in range(self.n_layers-1):
                weight = self.random_init((self.layer_sizes[i],self.layer_sizes[i+1]))
                self.weights.append(weight) 
        elif(weight_init=="normal"):
            for i in range(self.n_layers-1):
                weight = self.normal_init((self.layer_sizes[i],self.layer_sizes[i+1]))
                self.weights.append(weight)
        else:
            raise Exception("Error in setting weights")

        for i in range(self.n_layers-1):
            bias = self.zero_init((1,self.layer_sizes[i+1]))
            self.biases.append(bias)        

    def relu(self, X):
        return X*(X>0)

    def relu_grad(self, X):
        return np.array(X>0,dtype=int)

    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def sigmoid_grad(self, X):
        return self.sigmoid(X)*(1-self.sigmoid(X))

    def linear(self, X):
        return X

    def linear_grad(self, X):
        return np.ones(X.shape)

    def tanh(self, X):
        return np.tanh(X)

    def tanh_grad(self, X):
        return 1 - np.tanh(X)**2

    def softmax(self, X):
        new_arr = []
        # print(type(X[0]))
        for i in X:
            # print(type(i))
            exponential = np.exp(i)
            total = exponential.sum()
            new_arr.append(exponential/total)
        return np.array(new_arr)

    def softmax_grad(self, X):
        return X*(1-X)

    def leaky_relu(self,z):
      return np.maximum(0.01 * z, z)
    def leaky_relu_gradient(self,z):
        grad = np.ones_like(z)
        grad[z < 0] = 0.01
        return grad

    def zero_init(self, shape):
        return np.zeros(shape)

    def random_init(self, shape):
        return np.random.rand(shape[0],shape[1])*0.01

    def normal_init(self, shape):
        return np.random.normal(size = shape)*0.01

    def activate(self, X):
        if(self.activation == "relu"):
            return self.relu(X)
        elif(self.activation == "sigmoid"):
            return self.sigmoid(X)
        elif(self.activation == "linear"):
            return self.linear(X)
        elif(self.activation == "tanh"):
            return self.tanh(X)
        elif(self.activation == "softmax"):
            return self.softmax(X)
        elif(self.activation=='leaky_relu'):
            return self.leaky_relu(X)
        else:
            print("error in activate fucntion")
    
    def activate_grad(self, X):
        if(self.activation == "relu"):
            return self.relu_grad(X)
        elif(self.activation == "sigmoid"):
            return self.sigmoid_grad(X)
        elif(self.activation == "linear"):
            return self.linear_grad(X)
        elif(self.activation == "tanh"):
            return self.tanh_grad(X)
        elif(self.activation == "softmax"):
            return self.softmax_grad(X)
        elif(self.activation=='leaky_relu'):
            return self.leaky_relu_gradient(X)
        else:
            print("error in activate fucntion grad")
    
    def cross_entropy(self, y_pred, y_true):
        ce =  -1*np.log(y_pred[np.arange(len(y_true)), y_true.argmax(axis=1)])
        return np.sum(ce)
    
    def forward(self, X):
        before_activation = []
        after_activation = []
        x = deepcopy(X)
        for i in range(self.n_layers-2):
            op = x.dot(self.weights[i]) + self.biases[i]
            before_activation.append(op)
            op = self.activate(op)
            after_activation.append(op)
            x = op
        op = x.dot(self.weights[-1]) + self.biases[-1]
        before_activation.append(op)
        op = self.softmax(op)
        after_activation.append(op)
        return before_activation, after_activation
    
    def backward(self, y, before_activation, after_activation):
        grads = []
        final_pred = after_activation[-1]
        loss = final_pred - y
        grads.append(loss)
        for layer in range(self.n_layers - 3, -1, -1):
            curr_error = loss.dot(self.weights[layer+1].T)
            grad = self.activate_grad(before_activation[layer])
            loss = curr_error*grad
            grads.append(loss)
        grads.reverse()
        return grads
    
    def fit(self, X, y, X_test=None, y_test=None):
        loss = []
        val_loss = []
        for epoch in range(self.num_epochs):
            for batch in range(0,len(X),self.batch_size):
                currX = X[batch:batch+self.batch_size,:]
                currY = y[batch:batch+self.batch_size,:]    
                bef,aft = self.forward(currX)
                grads = self.backward(currY, bef, aft)
                zumm = currX
                for i in range(self.n_layers-1):
                    grad = zumm.T.dot(grads[i])/len(currX)
                    zumm = aft[i]
                    self.weights[i] = self.weights[i] - self.learning_rate*grad
                    self.biases[i] = self.biases[i] - self.learning_rate*np.sum(grads[i],axis=0)/len(currX)
            #cross entropy
            b,a = self.forward(X)
            loss.append(self.cross_entropy(a[-1],y)/len(y))
            if(loss[-1]<self.min_loss):
                self.min_loss = loss[-1]
            b,a = self.forward(X_test)
            val_loss.append(self.cross_entropy(a[-1],y_test)/len(y_test))
            print("epoch",epoch,", loss:",loss[-1])
            if(self.convergence != None):
                if((loss[-1] - self.min_loss > 0.1)):
                    print("Stopping iteration due to convergence (minima lost)")
                    break
                if(len(loss)>2 and epoch > self.num_epochs//5):
                    if(abs(loss[-2] - loss[-1]) < self.convergence):
                        print("Stopping iteration due to convergence")
                        break
        self.loss = loss
        self.val_loss = val_loss
        return self

    def predict_proba(self, X):
        return self.forward(X)[1][-1]

    def predict(self, X):
        return self.forward(X)[1][-1].argmax(axis=1)
          
    def score(self, X, y):
        y_pred = self.predict(X)
        c = 0
        for i in range(len(y_pred)):
            if(y[i][y_pred[i]]==1):
                    c+=1
        return c/len(y_pred)