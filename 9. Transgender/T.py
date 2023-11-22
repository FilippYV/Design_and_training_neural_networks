import numpy as np


# Функция активации ReLU
def relu(x):
    return np.maximum(0, x)


# Функция softmax для вычисления весов внимания
def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# Механизм внимания - scaled dot-product attention
def scaled_dot_product_attention(Q, K, V):
    dk = K.shape[-1]
    # Шаг 1: Вычисление весов внимания
    attention_scores = np.matmul(Q, K.T) / np.sqrt(dk)
    attention_weights = softmax(attention_scores, axis=-1)

    # Шаг 2: Взвешенная сумма значений
    attention_output = np.matmul(attention_weights, V)

    return attention_output, attention_weights


# Многослойный перцептрон (MLP)
def feed_forward(x, w1, b1, w2, b2):
    h = np.dot(x, w1) + b1
    h_relu = relu(h)
    output = np.dot(h_relu, w2) + b2
    return output


# Полная модель трансформера
def transformer_model(Q, K, V, w1, b1, w2, b2):
    attention_output, attention_weights = scaled_dot_product_attention(Q, K, V)
    mlp_output = feed_forward(attention_output, w1, b1, w2, b2)
    return mlp_output, attention_weights


# Функция обучения
def train(X, y, learning_rate=0.01, epochs=100):
    input_dim = X.shape[-1]
    hidden_dim = 32
    output_dim = y.shape[-1]

    # Инициализация весов для механизма внимания и MLP
    w1 = np.random.randn(hidden_dim, hidden_dim)
    b1 = np.zeros(hidden_dim)
    w2 = np.random.randn(hidden_dim, output_dim)
    b2 = np.zeros(output_dim)

    for epoch in range(epochs):
        for i in range(len(X)):
            Q = K = V = X[i]  # В упрощенной версии принимаем одинаковые Q, K, V
            target = y[i]

            # Прямое распространение
            output, _ = transformer_model(Q, K, V, w1, b1, w2, b2)

            # Вычисление ошибки и градиентов
            loss = np.mean((output - target) ** 2)
            d_output = 2 * (output - target)

            # Обратное распространение ошибки
            d_output_w2 = np.dot(relu(np.dot(Q, w1) + b1).T, d_output)
            d_output_b2 = np.sum(d_output, axis=0)
            d_output_hidden = np.dot(d_output, w2.T)
            d_output_hidden_relu = (np.dot(Q, w1) + b1 > 0) * d_output_hidden
            d_output_w1 = np.dot(X.T, d_output_hidden_relu)
            d_output_b1 = np.sum(d_output_hidden_relu, axis=0)

            # Обновление весов
            w1 -= learning_rate * d_output_w1
            b1 -= learning_rate * d_output_b1
            w2 -= learning_rate * d_output_w2
            b2 -= learning_rate * d_output_b2

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return w1, b1, w2, b2


# Функция предсказания
def predict(X, w1, b1, w2, b2):
    predictions = []
    for i in range(len(X)):
        Q = K = V = X[i]
        output, _ = transformer_model(Q, K, V, w1, b1, w2, b2)
        predictions.append(output)
    return np.array(predictions)


def prepare_data(X):
    # Создание матрицы для представления слов в последовательностях
    embedding_dim = 50  # Размерность эмбеддинга
    vocab_size = len(set(np.concatenate(X)))  # Размер словаря
    embeddings = np.random.randn(vocab_size, embedding_dim)

    # Преобразование слов в эмбеддинги
    embedded_X = [embeddings[sequence] for sequence in X if sequence < vocab_size]

    return np.array(embedded_X)



text_data = ["я", "люблю", "гулять", "по", "парку", "собакой"]
word_to_index = {word: idx for idx, word in enumerate(set(text_data))}
index_to_word = {idx: word for word, idx in word_to_index.items()}

sequence_length = 5
input_sequences = []
target_sequences = []

for i in range(len(text_data) - sequence_length):
    input_seq = text_data[i:i + sequence_length]
    target_seq = text_data[i + 1:i + sequence_length + 1]

    input_sequences.append([word_to_index[word] for word in input_seq])
    target_sequences.append([word_to_index[word] for word in target_seq])

X = np.array(input_sequences)
y = np.array(target_sequences)

# Подготовка данных в матрицы для Q, K, V
embedded_X = prepare_data(X)
Q = K = V = embedded_X

# Обучение модели
trained_weights = train(Q, y, learning_rate=0.01, epochs=100)

# Предсказание
predictions = predict(Q, *trained_weights)

# Вывод результатов
for i, pred in enumerate(predictions):
    input_words = ' '.join([index_to_word[idx] for idx in X[i]])
    target_words = ' '.join([index_to_word[idx] for idx in y[i]])
    pred_words = ' '.join([index_to_word[np.argmax(p)] for p in pred])

    print(f"Input: {input_words} | Target: {target_words} | Prediction: {pred_words}")
