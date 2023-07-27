# Multi-Layer-Perceptron--MLP--Neural-Network-from-Scratch
- For the full project description, please take a look at Project_Description.pdf
- This program implements a multi-layer perceptron (MLP) neural network scratch using Python, i.e. **without** the use of machine learning libraries and external libraries other than NumPy. It was used to classify from four different datasets: Spiral, Circle, XOR, Gaussian (training the model on each dataset took less than 2 minutes); I achieved 94%+ accuracy on each of the sample datasets provided in course grading.
- My program name is NeuralNetwork3.py; the neural network consists of 4 input nodes, 2 output nodes (utilizing soft max), and 1 hidden layer with 10 nodes. More details in the program file. 

Notes:
- The program takes in 3 input files, for example:
   python3 NeuralNetwork3.py spiral_train_data.csv spiral_train_label.csv spiral_test_data.csv
- The program outputs the predictions of the model in a csv file: test_predictions.csv