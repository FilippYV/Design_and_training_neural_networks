import numpy as np
import pandas as pd

data = pd.read_csv('../Datasets/Daily-minimum-temperatures/daily-minimum-temperatures-in-me.csv')
data = data.drop('Date', axis=1).to_numpy()
data = data.flatten()
new_data = []
for i in data:
    new_data.append(float(i.replace('?', '')))

maximum = max(new_data)

print('maximum', maximum)

# Нормализация данных
new_data = np.array(new_data) / maximum

# Разделение на подмассивы по 10 элементов
data = np.array_split(new_data, len(new_data) / 10)
del data[-1]
print('data', data, '\n')

# Подготовка обучающих данных
data = np.array(data)
X = data[:, :-1]  # Входные последовательности (все числа, кроме последнего)
y = data[:, -1]  # Выходные числа (следующее число после входной последовательности)
print('data', data[1])
print('X', X[1])
print('y', y[1])

# Гиперпараметры модели
input_size = 1  # Размерность входных данных
hidden_size = 64  # Размер скрытого слоя
output_size = 1  # Размер выходного слоя
learning_rate = 0.001
num_epochs = 100

# Инициализация весов
weights = {
    'Wq': np.random.randn(input_size, hidden_size),
    'Wk': np.random.randn(input_size, hidden_size),
    'Wv': np.random.randn(input_size, hidden_size),
    'Wo': np.random.randn(hidden_size, output_size),
    'bq': np.zeros((1, hidden_size)),
    'bk': np.zeros((1, hidden_size)),
    'bv': np.zeros((1, hidden_size)),
    'bo': np.zeros((1, output_size))
}


# Функции активации и градиенты
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Обучение трансформера
for epoch in range(num_epochs):
    loss = 0

    for i in range(len(X)):
        # Прямое распространение (forward pass)
        input_seq = X[i].reshape(-1, 1)
        target = y[i]

        # Attention
        q = np.dot(input_seq, weights['Wq']) + weights['bq']
        k = np.dot(input_seq, weights['Wk']) + weights['bk']
        v = np.dot(input_seq, weights['Wv']) + weights['bv']

        scores = np.dot(q, k.T) / np.sqrt(hidden_size)
        attention_weights = np.dot(scores, v)

        # Attention Layer
        attention_output = np.sum(attention_weights, axis=0, keepdims=True)

        # Output Layer
        output = np.dot(attention_output, weights['Wo']) + weights['bo']
        prediction = sigmoid(output)

        # Loss
        loss += np.square(prediction - target)

        # Обратное распространение (backpropagation) для обновления весов
        d_output = 2 * (prediction - target) * sigmoid_derivative(prediction)

        # Обратное распространение attention
        d_attention_output = np.dot(d_output, weights['Wo'].T)
        d_attention_weights = d_attention_output * np.ones_like(attention_weights) / len(attention_weights)

        d_v = np.dot(scores.T, d_attention_weights)
        d_scores = np.dot(d_attention_weights, v.T)
        d_k = np.dot(d_scores.T, q)  # Используйте транспонированную матрицу d_scores для умножения на q
        d_q = np.dot(d_scores.T, k)  # Используйте транспонированную матрицу d_scores для умножения на k

        d_weights = {
            'Wo': np.dot(attention_output.T, d_output),
            'bo': np.sum(d_output, axis=0, keepdims=True),
            'Wv': np.dot(input_seq.T, d_v),
            'bv': np.sum(d_v, axis=0, keepdims=True),
            'Wk': np.dot(input_seq.T, d_k),
            'bk': np.sum(d_k, axis=0, keepdims=True),
            'Wq': np.dot(input_seq.T, d_q),
            'bq': np.sum(d_q, axis=0, keepdims=True)
        }

        # Обновление весов
        for weight_name in weights:
            weights[weight_name] -= learning_rate * d_weights.get(weight_name, 0)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss / len(X)}')

# Предсказание для новой последовательности чисел
# 10.7, 8.2, 6.1, 4.5, 6.1, 9.8, 9.7, 8.2, 8.4, 8.5
test_sequence = np.array([10.7, 8.2, 6.1, 4.5, 6.1, 9.8, 9.7, 8.2, 8.4])
test_sequence = test_sequence / maximum
predicted_value = None

# Прямое распространение по обученным весам
for i in range(len(test_sequence)):
    input_seq = test_sequence[i].reshape(-1, 1)

    q = np.dot(input_seq, weights['Wq']) + weights['bq']
    k = np.dot(input_seq, weights['Wk']) + weights['bk']
    v = np.dot(input_seq, weights['Wv']) + weights['bv']

    scores = np.dot(q, k.T) / np.sqrt(hidden_size)
    attention_weights = np.dot(scores, v)

    attention_output = np.sum(attention_weights, axis=0, keepdims=True)
    output = np.dot(attention_output, weights['Wo']) + weights['bo']
    predicted_value = sigmoid(output)
predicted_value = predicted_value[0][0]
print(f'Predicted value: {predicted_value}')
print(f'Predicted value: {predicted_value * maximum}')
