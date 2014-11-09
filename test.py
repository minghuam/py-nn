import MLP
import activation_function
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#nn = MLP.MLP([784, 20, 10], activation_function.sigmoid)
#nn.SGD(training_data, 10, 10, 3.0, test_data)

nn = MLP.MLP([784, 20, 10], activation_function.tanh)
nn.SGD(training_data, 10, 10, 0.3, test_data)
