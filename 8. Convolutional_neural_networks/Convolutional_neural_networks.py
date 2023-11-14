import numpy as np
import pandas as pd


def get_train_data():
    data_train = pd.read_csv('../Datasets/Fashion MNIST/fashion-mnist_train.csv')
    X_train = data_train.drop(['label'], axis=1).to_numpy()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))  # Обновленный формат

    y_train = data_train['label'].to_numpy()
    # print('x_train\n', X_train[:5])
    # print()
    # print('y_train\n', y_train[:5])
    return X_train, y_train


class CNN:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes
        self.filters1 = np.random.randn(3, 3, 1, 32)
        self.filters2 = np.random.randn(2, 2, 32, 64)  # Уменьшение размера фильтров второго сверточного слоя
        self.weights_fc = np.random.randn(7 * 7 * 64, 100)
        self.weights_output = np.random.randn(100, num_classes)

    def convolve2d(self, input_data, filters):
        _, height, width, depth = input_data.shape
        filter_height, filter_width, input_depth, num_filters = filters.shape

        output_height = height - filter_height + 1
        output_width = width - filter_width + 1

        output_data = np.zeros((output_height, output_width, num_filters))

        for i in range(output_height):
            for j in range(output_width):
                # Используйте расширение осей для корректного выполнения операции свертки
                output_data[i, j, :] = np.sum(
                    input_data[:, i:i + filter_height, j:j + filter_width, np.newaxis] * filters,
                    axis=(1, 2, 3))

        return output_data

    def max_pooling(self, input_data, pool_size=(2, 2), padding='valid'):
        height, width, num_filters = input_data.shape
        pool_height, pool_width = pool_size

        if padding == 'valid':
            output_height = height // pool_height
            output_width = width // pool_width
        elif padding == 'same':
            output_height = height
            output_width = width
        else:
            raise ValueError("Invalid padding mode. Use 'valid' or 'same'.")

        output_data = np.zeros((output_height, output_width, num_filters))

        for i in range(output_height):
            for j in range(output_width):
                output_data[i, j, :] = np.max(
                    input_data[i * pool_height:(i + 1) * pool_height, j * pool_width:(j + 1) * pool_width, :],
                    axis=(0, 1))

        return output_data

    def flatten(self, input_data):
        return input_data.reshape(-1)

    def fully_connected(self, input_data, weights):
        return np.dot(input_data, weights)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def train(self, X_train, y_train, learning_rate=0.01, epochs=10):
        for epoch in range(epochs):
            for index, image in enumerate(X_train):
                # Расширение размерности для корректного проведения операции свертки
                image = image.reshape((1, 28, 28, 1))
                conv1_output = self.convolve2d(image, self.filters1)
                pool1_output = self.max_pooling(conv1_output)

                # Reshape the pool1_output tensor to have a shape of (height, width, 1, num_filters)
                pool1_output = pool1_output.reshape(
                    (pool1_output.shape[0], pool1_output.shape[1], 1, pool1_output.shape[2]))

                conv2_output = self.convolve2d(pool1_output, self.filters2)
                pool2_output = self.max_pooling(conv2_output)


                flattened_output = self.flatten(pool2_output)

                print("pool1_output shape:", pool1_output.shape)
                print("pool2_output shape:", pool2_output.shape)
                print("flattened_output shape:", flattened_output.shape)
                # print("fc_output shape:", fc_output.shape)

                fc_output = self.fully_connected(flattened_output, self.weights_fc)
                fc_output_relu = np.maximum(0, fc_output)


                output = self.fully_connected(fc_output_relu, self.weights_output)
                predicted_probs = self.softmax(output)

                loss_gradient = predicted_probs - y_train[index]
                weights_output_gradient = np.outer(fc_output_relu, loss_gradient)
                fc_output_gradient = np.dot(self.weights_output, loss_gradient)
                fc_output_gradient[fc_output < 0] = 0
                weights_fc_gradient = np.outer(flattened_output, fc_output_gradient)

                self.weights_output -= learning_rate * weights_output_gradient
                self.weights_fc -= learning_rate * weights_fc_gradient.reshape(self.weights_fc.shape)

    def predict(self, X_test):
        predictions = []
        for image in X_test:
            # Расширение размерности для корректного проведения операции свертки
            image = image.reshape((1, 28, 28, 1))
            conv1_output = self.convolve2d(image, self.filters1)
            pool1_output = self.max_pooling(conv1_output)

            conv2_output = self.convolve2d(pool1_output, self.filters2)
            pool2_output = self.max_pooling(conv2_output)

            flattened_output = self.flatten(pool2_output)
            fc_output = self.fully_connected(flattened_output, self.weights_fc)
            fc_output_relu = np.maximum(0, fc_output)

            output = self.fully_connected(fc_output_relu, self.weights_output)
            predictions.append(output)

        return predictions


if __name__ == '__main__':
    X_train, y_train = get_train_data()
    num_classes = 10
    print(X_train.shape)
    # simple_conv_net = CNN(input_size=X_train.shape, num_classes)
    # simple_conv_net.train(X_train, y_train)
    # predictions = simple_conv_net.predict(X_test)
