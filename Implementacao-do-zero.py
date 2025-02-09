import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist

# Função para converter rótulos em one-hot encoding sem bibliotecas externas
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    return one_hot

# Funções de ativação
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Carregamento e pré-processamento dos dados
def carregar_e_preprocessar_dados():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    y_train_onehot = one_hot_encode(y_train, 10)
    y_test_onehot = one_hot_encode(y_test, 10)

    return X_train, y_train_onehot, X_test, y_test_onehot, y_test

# Inicialização dos pesos e biases para 2 camadas ocultas
def inicializar_pesos(input_size, hidden_size1, hidden_size2, output_size):
    W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size1))
    W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
    b2 = np.zeros((1, hidden_size2))
    W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
    b3 = np.zeros((1, output_size))
    return W1, b1, W2, b2, W3, b3

# Forward propagation com 2 camadas ocultas
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = softmax(z3)
    return a1, a2, a3

# Função de custo
def calcular_custo(y, y_hat):
    m = y.shape[0]
    epsilon = 1e-8
    cost = (-1/m) * np.sum(y * np.log(y_hat + epsilon))
    return cost

# Backpropagation para 2 camadas ocultas
def backpropagation(X, y, a1, a2, a3, W1, W2, W3):
    m = X.shape[0]

    dz3 = a3 - y
    dW3 = (1/m) * np.dot(a2.T, dz3)
    db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)

    dz2 = np.dot(dz3, W3.T) * (a2 > 0)
    dW2 = (1/m) * np.dot(a1.T, dz2)
    db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * (a1 > 0)
    dW1 = (1/m) * np.dot(X.T, dz1)
    db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

# Treinamento do modelo com 2 camadas ocultas
def treinar_modelo(X_train, y_train, X_test, y_test, hidden_size1, hidden_size2, learning_rate, epochs, batch_size):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    W1, b1, W2, b2, W3, b3 = inicializar_pesos(input_size, hidden_size1, hidden_size2, output_size)

    custos_treinamento = []
    custos_teste = []

    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            a1, a2, a3 = forward_propagation(X_batch, W1, b1, W2, b2, W3, b3)
            dW1, db1, dW2, db2, dW3, db3 = backpropagation(X_batch, y_batch, a1, a2, a3, W1, W2, W3)

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W3 -= learning_rate * dW3
            b3 -= learning_rate * db3

        _, _, y_hat_train = forward_propagation(X_train, W1, b1, W2, b2, W3, b3)
        custo_treinamento = calcular_custo(y_train, y_hat_train)
        custos_treinamento.append(custo_treinamento)

        _, _, y_hat_test = forward_propagation(X_test, W1, b1, W2, b2, W3, b3)
        custo_teste = calcular_custo(y_test, y_hat_test)
        custos_teste.append(custo_teste)

        print(f"Época {epoch+1}/{epochs}, Custo Treinamento: {custo_treinamento:.4f}, Custo Teste: {custo_teste:.4f}")

    return W1, b1, W2, b2, W3, b3, custos_treinamento, custos_teste

# Avaliação do modelo
def avaliar_modelo(X_test, y_test, W1, b1, W2, b2, W3, b3):
    _, _, y_hat_test = forward_propagation(X_test, W1, b1, W2, b2, W3, b3)
    y_pred = np.argmax(y_hat_test, axis=1)
    y_true = np.argmax(y_test, axis=1)

    conf_matrix = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_true, y_pred):
        conf_matrix[true, pred] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Real")
    plt.title("Matriz de Confusão")
    plt.show()

    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    support = np.sum(conf_matrix, axis=1)

    print("\nPrecision por classe:", precision)
    print("Recall por classe:", recall)
    print("F1-score por classe:", f1_score)
    print("Support por classe:", support)

    accuracy = np.mean(y_pred == y_true)
    print(f"\nAcurácia nos dados de teste: {accuracy:.4f}")

# Carregamento e execução
# Com 128 na segunda camada oculta e 50 epochs, a execução ocorreu em um tempo consideravel, não muito demorado e teve uma acuracia de
# 97,65%.
# Com 128 na segunda camada oculta e 40 epochs, demorou consideravelmente e teve 97,61% de acuracia, e com 45 epochs
# e 256 na camada oculta, demorou bastante e teve uma acuracia de 97,79%´, porem a partir as 43 epochs o modelo começa a iniciar um overfitting.
# Com 256 na segunda camada e 50 epochs ocorre overfitting, pois enquanto os dados de treino caem, os de teste aumentam.
# Com 128 na segunda camada e 60 epochs, ele passa a ter overfitting.
# com 530 na primeira camada, 256 na segunda e 30 epochs, ele demora para executar, porém não ocorre overfitting, alem de ter uma acuracia levemente menor apenas, com 97,48%
# Resumindo, até então o melhor resultado foi obtido com 530 hidden1 256 hidden2 e 30 epochs, pois embora a acuracia tenha caido em apenas 0,4% (irrelevante)
# O modelo está sem overfitting e com um melhor balanceamento de classes pelas metricas de f1-score, precision e recall.
X_train, y_train, X_test, y_test, y_test_labels = carregar_e_preprocessar_dados()
W1, b1, W2, b2, W3, b3, _, _ = treinar_modelo(X_train, y_train, X_test, y_test, 530, 256, 0.01, 30, 64)
avaliar_modelo(X_test, y_test, W1, b1, W2, b2, W3, b3)
