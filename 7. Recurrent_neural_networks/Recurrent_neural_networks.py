import numpy as np

class RNN:
    def __init__(self, learning_rate, epochs, hidden_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.Wh_h = np.random.randn(hidden_size, hidden_size)
        self.Wh_x = np.random.randn(hidden_size, 1)
        self.bh = np.zeros((hidden_size, 1))
        self.mean = None
        self.std = None

    def normalize(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean

    def train(self, bitcoin_prices):
        bitcoin_prices_normalized = self.normalize(bitcoin_prices)
        for epoch in range(self.epochs):
            loss = 0
            h_prev = 0

            for i in range(len(bitcoin_prices_normalized) - 1):
                # Прямой проход
                h = np.tanh(np.dot(self.Wh_h, h_prev) + np.dot(self.Wh_x, bitcoin_prices_normalized[i]) + self.bh)

                # Предсказание
                predicted_price = np.dot(self.Wh_x, bitcoin_prices_normalized[i + 1])

                # Вычисление ошибки
                loss += (predicted_price - bitcoin_prices_normalized[i + 1]) ** 2

                # Обратный проход
                dWxh = np.dot((predicted_price - bitcoin_prices_normalized[i + 1]), bitcoin_prices_normalized[i + 1])
                dWhh = np.dot((predicted_price - bitcoin_prices_normalized[i + 1]), h_prev)
                dbh = (predicted_price - bitcoin_prices_normalized[i + 1])

                # Обновление весов
                self.Wh_x -= self.learning_rate * dWxh
                self.Wh_h -= self.learning_rate * dWhh
                self.bh -= self.learning_rate * dbh

                h_prev = h

            # Вывод ошибки на каждой эпохе
            print(f'Epoch {epoch + 1}, Loss: {loss}')

    def predict(self, bitcoin_prices):
        bitcoin_prices_normalized = self.normalize(bitcoin_prices)
        h_prev = 0
        for i in range(len(bitcoin_prices_normalized) - 1):
            h = np.tanh(np.dot(self.Wh_h, h_prev) + np.dot(self.Wh_x, bitcoin_prices_normalized[i]) + self.bh)
            h_prev = h

        next_price_normalized = np.tanh(np.dot(self.Wh_h, h) + np.dot(self.Wh_x, bitcoin_prices_normalized[-1]) + self.bh)
        return self.denormalize(next_price_normalized)


rnn = RNN(learning_rate=0.01, epochs=50, hidden_size=1)
bitcoin_prices = np.array([100, 108, 110, 115, 105, 112, 120, 125, 135, 140])
rnn.train(bitcoin_prices)

next_price = rnn.predict([140, 145, 155, 160])
print(f'Predicted next Bitcoin price: {next_price}')
