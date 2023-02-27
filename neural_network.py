import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork :
    def __init__ (self, learning_rate = 0.10):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_der(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    
    
    def predict(self, input_vector):
        layer1 = np.dot(input_vector, self.weights) + self.bias
        layer2 = self._sigmoid(layer1)

        prediction = layer2
        return prediction
    
    def _compute_gradient(self, input_vector, target):
        layer1 = np.dot(input_vector, self.weights) + self.bias
        layer2 = self._sigmoid(layer1)
        
        prediction = layer2
        
        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_der(layer1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        
        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        
        return derror_dbias, derror_dweights
        
    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)
        
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        
        for current_iteration in range(iterations):
            # pick a data instance at random 
            random_index_data = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_index_data]
            target = targets[random_index_data]
            
            # compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradient(input_vector, target)
            
            self._update_parameters(derror_dbias, derror_dweights)
            
            # measure the cumulative error for all the instances
            if (current_iteration % 100 == 0):
                cumulative_error = 0
                # loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target_point = targets[data_instance_index]
                    
                    prediction = self.predict(data_point)
                    error = np.square(prediction - target_point)
                    
                    cumulative_error = cumulative_error + error
                
                cumulative_errors.append(cumulative_error)
                    
        return cumulative_errors        
                
                
   
input_vectors = np.array(
    [
        [3, 1.5],
        [2, 1],
        [4, 1.5],
        [3, 4],
        [3.5, 0.5],
        [2, 0.5],
        [5.5, 1],
        [1, 1],
    ]
)             
targets = np.array([1, 0, 1, 1, 0, 0, 1, 1])
learning_rate = 0.1
            
import os, psutil 
process = psutil.Process(os.getpid())
            
print("create neuralnetwork")
neural_nw = NeuralNetwork(learning_rate)



print("prediction BEFORE training")
for data_instance_index in range(len(input_vectors)):
    data_point = input_vectors[data_instance_index]
    target_point = targets[data_instance_index]
    
    prediction = neural_nw.predict(data_point)
    
    if (prediction < 0.50):
        pr_r = 0
    elif (prediction >= 0.5):
        pr_r = 1
    print(f"input array: {data_point}; prediction: {prediction}; target: {target_point}; pred01: {pr_r}; result: {pr_r - target_point}")
    


print("start training error")
training_error = neural_nw.train(input_vectors, targets, 50000)

print("create error plot")
plt.plot(training_error)
plt.xlabel("iterations")
plt.ylabel("error for all training instances")
plt.savefig("cumulative_error.png")


print("prediction after training")
for data_instance_index in range(len(input_vectors)):
    data_point = input_vectors[data_instance_index]
    target_point = targets[data_instance_index]
    
    prediction = neural_nw.predict(data_point)
    
    if (prediction < 0.50):
        pr_r = 0
    elif (prediction >= 0.5):
        pr_r = 1
    print(f"input array: {data_point}; prediction: {prediction}; target: {target_point}; pred01: {pr_r}; result: {pr_r - target_point}")
    

print("memory usage by python program (Mib): "+str(process.memory_info().rss / 1024**2))
