import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder


def get_train_data():
    data_train = pd.read_csv('../Datasets/Fashion MNIST/fashion-mnist_train.csv')
    X_train = data_train.drop(['label'], axis=1).to_numpy()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

    y_train = data_train['label'].to_numpy()

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))

    return X_train, y_train_onehot


class CNN:
    def __init__(self, input_size, num_classes, count_convolutional_layer):
        self.input_size = input_size
        self.num_classes = num_classes
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
        weight_count_convolutional = [[0.1, 0.1, 0.1] for
                                      i in range(3)]
        return weight_count_convolutional

    def flatten(self, input_data):
        return input_data.reshape(-1)

    def convolve2d(self, input_data):
        height, width = input_data.shape
        mask = np.zeros((height + 2, width + 2))
        mask[1:height + 1, 1:height + 1] = input_data
        output_data = []
        for i in range(1, height + 1):
            s_array = []
            for j in range(1, width + 1):
                summ = 0
                for ii in range(-1, 2):
                    for jj in range(-1, 2):
                        summ += mask[i + ii][j + jj] * self.weight_count_convolutional[ii + 1][jj + 1]
                s_array.append(summ)
            output_data.append(s_array)
        return output_data

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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

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
                # print('fully_connected', fully_connected)
                output = self.sigmoid(fully_connected)
                # print('output', output)
                # print('y[index_training_example]', y[index_training_example])
                # Backward pass
                loss_gradient = output - y[index_training_example]
                fc_output_gradient = np.dot(self.flatten_weight.T, loss_gradient)
                fc_output_gradient[fc_output_gradient < 0] = 0

                # Update weights
                self.flatten_weight -= learning_rate * np.outer(loss_gradient, flattened_output)

                # Optionally, update convolutional layer weights here

                # Print or log the loss
                if index_training_example % 1000 == 0:
                    current_loss = np.mean(-y_train * np.log(output) - (1 - y_train) * np.log(1 - output))
                    print(f'Epoch {eph}, Example {index_training_example}, Loss: {current_loss}')


if __name__ == '__main__':
    X_train, y_train = get_train_data()
    simple_conv_net = CNN(
        input_size=X_train[0].shape[0],
        num_classes=len(np.unique(y_train)),
        count_convolutional_layer=1)
    simple_conv_net.train(X=X_train, y=y_train, epoch=1)
