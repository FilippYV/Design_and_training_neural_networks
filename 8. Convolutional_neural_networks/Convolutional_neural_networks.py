import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
def get_train_data():
    data_train = pd.read_csv('../Datasets/Fashion MNIST/fashion-mnist_train.csv')
    X_train = data_train.drop(['label'], axis=1).to_numpy()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

    y_train = data_train['label'].to_numpy()

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))

    return X_train, y_train_onehot
#%%
def print_accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    return predictions
class CNN:
    def __init__(self, input_size, num_classes, count_convolutional_layer):
        self.input_size = input_size
        self.num_classes = num_classes  # Update the initialization
        self.count_convolutional_layer = count_convolutional_layer
        self.weight_count_convolutional = self.get_weight_count_convolutional()
        self.flatten_weight = self.get_flatten_weight()

    def get_flatten_weight(self):
        flatten_weight = []
        for i in range(self.num_classes):
            mask = []
            for j in range(49):
                mask.append(random.uniform(0, 1))
            flatten_weight.append(mask)
        return np.array(flatten_weight)

    def get_weight_count_convolutional(self):
        return np.random.uniform(0, 1, size=(3, 3))

    def flatten(self, input_data):
        return input_data.reshape(-1)

    def convolve2d(self, input_data):
        channels = 1  # Default value for single channel
        if len(input_data.shape) == 2:
            height, width = input_data.shape
            input_data = input_data.reshape((height, width, 1))
        elif len(input_data.shape) == 3:
            height, width, channels = input_data.shape
        else:
            raise ValueError("Input data should be either 2D or 3D")

        mask = np.zeros((height + 2, width + 2, channels))
        mask[1:height + 1, 1:width + 1, :] = input_data
        output_data = []
        for i in range(1, height + 1):
            s_array = []
            for j in range(1, width + 1):
                summ = 0
                for ii in range(-1, 2):
                    for jj in range(-1, 2):
                        summ += np.sum(mask[i + ii, j + jj, :] * self.weight_count_convolutional[ii + 1][jj + 1])
                s_array.append(summ)
            output_data.append(s_array)
        return np.array(output_data)

    def max_pooling(self, data):
        leng = len(data)
        i = 0
        j = 0
        result = []
        while i < leng:
            result_str = []
            while j < leng:
                result_str.append(max([data[i][j], data[i][j + 1], data[i + 1][j], data[i + 1][j + 1]]))
                j += 2
            i += 2
            j = 0
            result.append(result_str)
        return np.array(result)

    def fully_connected(self, flattened_output):
        return np.dot(self.flatten_weight, flattened_output)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def train(self, X, y, epoch, learning_rate=0.01):
        for eph in range(epoch):
            for index_training_example, training_example in enumerate(X):
                training_example = np.array(training_example)
                conv1_output = self.convolve2d(training_example)
                pool1_output = self.max_pooling(conv1_output)

                conv2_output = self.convolve2d(pool1_output)
                pool21_output = self.max_pooling(conv2_output)

                flattened_output = self.flatten(pool21_output)
                fully_connected = self.fully_connected(flattened_output)
                output = self.sigmoid(fully_connected)
                # Backward pass
                loss_gradient = output - y[index_training_example]
                fc_output_gradient = np.dot(self.flatten_weight.T, loss_gradient)
                fc_output_gradient[fc_output_gradient < 0] = 0

                # Update weights
                self.flatten_weight -= learning_rate * np.outer(loss_gradient, flattened_output)

                # Optionally, update convolutional layer weights here

                # Print or log the loss
                if index_training_example % 1000 == 0:
                    current_loss = np.mean(-y_train * np.log(self.softmax(output)) - (1 - y_train) * np.log(1 - self.softmax(output)))
                    print(f'Epoch {eph}, Example {index_training_example}, Loss: {current_loss}')

    # Add this method to get the predicted probabilities using the trained model
    def predict_probabilities(self, X):
        predictions = []
        for example in X:
            conv1_output = self.convolve2d(example)
            pool1_output = self.max_pooling(conv1_output)

            conv2_output = self.convolve2d(pool1_output)
            pool2_output = self.max_pooling(conv2_output)

            flattened_output = self.flatten(pool2_output)
            fully_connected = self.fully_connected(flattened_output)
            output = self.softmax(fully_connected)
            predictions.append(output)

        return np.array(predictions)

# ... (rest of your existing code)

# Inside the main block
if __name__ == '__main__':
    X_train, y_train = get_train_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    simple_conv_net = CNN(
        input_size=X_train[0].shape[0],
        num_classes=y_train.shape[1],
        count_convolutional_layer=1)

    # Training
    simple_conv_net.train(X=X_train, y=y_train, epoch=1)

    # Testing
    print_accuracy(simple_conv_net, X_test, y_test)
