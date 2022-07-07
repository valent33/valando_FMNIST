*522841 - Valentin PORCHET, 522759 - William MAILLET (Erasmus students)*

# **Neural Networks - PV021**

### **Running the program**

IMPORTANT NOTE: If you do not want to use the RUN file, make sure you only use a specific number of threads before lauching the program, otherwise the whole capacity of the computer is going to be used. To do so, run the command `export OMP_NUM_THREADS=x` with `x` the number of threads you want.

To run the program, you can use the ```./RUN``` (if the error is "bad interpreter", you can try the command ```sed -i -e 's/\r$//' RUN```). If it does not work, you can compile the program using the following command (while being in the root folder of the project):

```
g++ -Wall -g -Ofast -Wextra ./src/activation.cpp ./src/init.cpp ./src/iostreams.cpp ./src/main.cpp ./src/matrix.cpp ./src/network.cpp ./src/utils.cpp -o network -lm -fopenmp
```

Then, just use the command:
<br>
```
./network
```

### **Output files**

The output files for the training predictions and testing predictions are respectively named `trainPredictions` and `testPredictions` and are located in the root folder of the project once the project finishes.


### **About**

This project is the implementation of a neural network trying to predict outputs using the Fashion MNIST dataset.
This is a feed-forward neural network in C, using the backpropagation. It has a single hidden layer of 250 neurons using the leakyReLU activation function and the output layer is made of 10 neurons with the softmax activation function.
The backpropagation uses the Adam optimizer.
Currently, its accuracy is around 89.5%.

