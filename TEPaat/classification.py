# load and manipulate functions
import pandas as pd
import numpy as np
# plot functions
import matplotlib.pyplot as plt
import seaborn as sns
# classifier function
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
# save model
import pickle

class Classifier():
    def __init__(self, features, labels, testratio = 0.2, state = 42):

        self.features = features
        self.labels = labels
        self.testratio = testratio
        self.state = state
        # split data
        self.trainXdata, self.testXdata, self.trainlabels, self.testlabel = train_test_split(self.features, self.labels, test_size = self.testratio, random_state = self.state)
    
    def fit_ANN(self, n_hidden = 10, learning_rate = 0.01, n_epochs = 1000, state = 42):
        # create algorithm instance
        self.model = MLPClassifier(hidden_layer_sizes = (n_hidden, ), max_iter = n_epochs, learning_rate_init = learning_rate, random_state = state)
        # fit data
        self.model.fit(self.trainXdata, self.trainlabels)
        # predict class
        self.ann_label_data = self.model.predict(self.testXdata)
        # print classification report
        print(classification_report(self.testlabel, self.ann_label_data))
        
        return self.ann_label_data

    def fit_SVC(self, C_ = 1.0, kernel_ = 'rbf'):
        # create algorithm instance
        self.model = SVC(C = C_, kernel = kernel_)
        # fit data
        self.model.fit(self.trainXdata, self.trainlabels)
        # predict class
        self.svc_label_data = self.model.predict(self.testXdata)
        # print classification report
        print(classification_report(self.testlabel, self.svc_label_data))
        
        return self.svc_label_data
   

    def confusion_matrix(self, fig_size = (6, 4)):
        """
        Tests the effect of increasing the number of hidden layers on model performance.
        Plots the confusion matrix.
        """
        plt.figure(figsize=fig_size)
        # Assuming you have already trained and tested your model and obtained the predicted labels
        y_pred = self.model.predict(self.testXdata)

        # Create the confusion matrix
        cm = confusion_matrix(self.testlabel, y_pred)

        # Create the heatmap using the confusion matrix
        sns.heatmap(cm, annot=True, cmap='Blues')

        # Set the axis labels and title
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


    def test_ANN_n_hidden(self, hidden_layers, learning_rate, n_epochs, state = 42):
        """
        Tests the effect of increasing the number of hidden layers on model performance.
        Plots the accuracy scores for each number of hidden layers.
        """
        accuracy_scores = []
        
        for n_hidden in hidden_layers:
            # Define the model
            model = MLPClassifier(hidden_layer_sizes=(n_hidden,), max_iter=n_epochs, learning_rate_init=learning_rate, random_state=state)
            
            # Train the model on the training data
            model.fit(self.trainXdata, self.trainlabels)
            
            # Make predictions on the test data
            y_pred = model.predict(self.testXdata)
            
            # Calculate the accuracy score and append to the list
            accuracy = accuracy_score(self.testlabel, y_pred)
            accuracy_scores.append(accuracy)
        
        # Plot the accuracy scores 
        x_pos = np.arange(len(hidden_layers))
        plt.bar(x_pos, accuracy_scores)
        plt.xticks(x_pos, hidden_layers)
        plt.xlabel('Number of Hidden Layers')
        plt.ylabel('Accuracy')
        plt.title('Effect of Number of Hidden Layers on Model Performance')
        plt.show()


    def test_ANN_learning_rate(self, hidden_layers, learning_rate, n_epochs, state = 42):
        """
        Tests the effect of increasing the learning rate on model performance.
        Plots the accuracy scores.
        """
        accuracy_scores = []
        
        for rate in learning_rate:
            # Define the model
            model = MLPClassifier(hidden_layer_sizes=(hidden_layers,), max_iter=n_epochs, learning_rate_init=rate, random_state=state)
            
            # Train the model on the training data
            model.fit(self.trainXdata, self.trainlabels)
            
            # Make predictions on the test data
            y_pred = model.predict(self.testXdata)
            
            # Calculate the accuracy score and append to the list
            accuracy = accuracy_score(self.testlabel, y_pred)
            accuracy_scores.append(accuracy)
        
        # Plot the accuracy scores
        x_pos = np.arange(len(learning_rate))
        plt.bar(x_pos, accuracy_scores)
        plt.xticks(x_pos, learning_rate)
        plt.xlabel('Learning rate')
        plt.ylabel('Accuracy')
        plt.title('Effect of learning rate on Model Performance')
        plt.show()


    def test_ANN_max_iter(self, hidden_layers, learning_rate, n_epochs, state = 42):
        """
        Tests the effect of increasing the maximum number of iteration on model performance.
        Plots the accuracy scores.
        """
        accuracy_scores = []
        
        for epoch in n_epochs:
            # Define the model 
            model = MLPClassifier(hidden_layer_sizes=(hidden_layers,), max_iter=epoch, learning_rate_init=learning_rate, random_state=state)
            
            # Train the model on the training data
            model.fit(self.trainXdata, self.trainlabels)
            
            # Make predictions on the test data
            y_pred = model.predict(self.testXdata)
            
            # Calculate the accuracy score and append to the list
            accuracy = accuracy_score(self.testlabel, y_pred)
            accuracy_scores.append(accuracy)
        
        # Plot the accuracy scores 
        x_pos = np.arange(len(n_epochs))
        plt.bar(x_pos, accuracy_scores)
        plt.xticks(x_pos, n_epochs)
        plt.xlabel('Maximum Number of Iteration')
        plt.ylabel('Accuracy')
        plt.title('Effect of max_iter on Model Performance')
        plt.show()


    def test_ANN_solver_algorithm(self, hidden_layers, learning_rate, n_epochs, algorithm, state = 42):
        """
        Tests the effect of solver types on model performance.
        Plots the accuracy scores.
        """
        accuracy_scores = []
        
        for method in algorithm:
            # Define the model 
            model = MLPClassifier(hidden_layer_sizes=(hidden_layers,), max_iter=n_epochs, solver = method, learning_rate_init=learning_rate, random_state=state)
            
            # Train the model on the training data
            model.fit(self.trainXdata, self.trainlabels)
            
            # Make predictions on the test data
            y_pred = model.predict(self.testXdata)
            
            # Calculate the accuracy score and append to the list
            accuracy = accuracy_score(self.testlabel, y_pred)
            accuracy_scores.append(accuracy)
        
        # Plot the accuracy scores 
        x_pos = np.arange(len(algorithm))
        plt.bar(x_pos, accuracy_scores)
        plt.xticks(x_pos, algorithm)
        plt.xlabel('Solver')
        plt.ylabel('Accuracy')
        plt.title('Effect of Solver on Model Performance')
        plt.show()  

    def test_SVC_kernel(self, kernel_, C_ = 1.0):
        """
        Tests the effect of kernel on model performance.
        Plots the accuracy scores.
        """
        accuracy_scores = []
        
        for ken in kernel_:
            # Define the model
            model = SVC(C = C_, kernel = ken)
            
            # Train the model on the training data
            model.fit(self.trainXdata, self.trainlabels)
            
            # Make predictions on the test data
            y_pred = model.predict(self.testXdata)
            
            # Calculate the accuracy score and append to the list
            accuracy = accuracy_score(self.testlabel, y_pred)
            accuracy_scores.append(accuracy)
        
        # Plot the accuracy scores 
        x_pos = np.arange(len(kernel_))
        plt.bar(x_pos, accuracy_scores)
        plt.xticks(x_pos, kernel_)
        plt.xlabel('Kernel')
        plt.ylabel('Accuracy')
        plt.title('Effect of Kernel on Model Performance')
        plt.show()  


    def test_SVC_Regularization(self, C_, kernel_ = 'rbf'):
        """
        Tests the effect of Regularization parameter on model performance.
        Plots the accuracy scores.
        """
        accuracy_scores = []
        
        for c in C_:
            # Define the model
            model = SVC(C = c, kernel = kernel_)
            
            # Train the model on the training data
            model.fit(self.trainXdata, self.trainlabels)
            
            # Make predictions on the test data
            y_pred = model.predict(self.testXdata)
            
            # Calculate the accuracy score and append to the list
            accuracy = accuracy_score(self.testlabel, y_pred)
            accuracy_scores.append(accuracy)
        
        # Plot the accuracy scores 
        x_pos = np.arange(len(C_))
        plt.bar(x_pos, accuracy_scores)
        plt.xticks(x_pos, C_)
        plt.xlabel('Regularization parameter')
        plt.ylabel('Accuracy')
        plt.title('Effect of Regularization parameter on Model Performance')
        plt.show()  

    def save_model(self, filename = "mlp_classifier"):
        """
        returns the fitted model.
        """
        # Save the model to a file using pickle
        pickle.dump(self.model, open(f'./models/{filename}_model.pkl', 'wb'))