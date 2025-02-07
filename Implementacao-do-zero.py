import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Funções de ativação
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Evita overflow e garante que a soma seja 1
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Carregamento e pré-processamento dos dados
def carregar_e_preprocessar_dados():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Redimensionamento
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    # One-hot encoding
    num_classes = 10
    y_train_onehot = to_categorical(y_train, num_classes)
    y_test_onehot = to_categorical(y_test, num_classes)

    return X_train, y_train_onehot, X_test, y_test_onehot

# Inicialização dos pesos
def inicializar_pesos(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return a1, a2

# Função de custo (entropia cruzada)
def calcular_custo(y, y_hat):
    m = y.shape[0]
    epsilon = 1e-8  # Valor pequeno para evitar log(0)
    cost = (-1/m) * np.sum(y * np.log(y_hat + epsilon))
    return cost

# Backpropagation
def backpropagation(X, y, a1, a2, W1, W2):
    m = X.shape[0]

    dz2 = a2 - y
    dW2 = (1/m) * np.dot(a1.T, dz2)
    db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

    dz1 = dz2.dot(W2.T) * (a1 > 0)  # Derivada da ReLU
    dW1 = (1/m) * np.dot(X.T, dz1)
    db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# Treinamento do modelo
def treinar_modelo(X_train, y_train, X_test, y_test, hidden_size, learning_rate, epochs, batch_size):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    W1, b1, W2, b2 = inicializar_pesos(input_size, hidden_size, output_size)

    custos_treinamento = []
    custos_teste = []

    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            a1, a2 = forward_propagation(X_batch, W1, b1, W2, b2)
            custo = calcular_custo(y_batch, a2)
            dW1, db1, dW2, db2 = backpropagation(X_batch, y_batch, a1, a2, W1, W2)

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        # Cálculo do custo nos dados de treinamento e teste
        _, y_hat_train = forward_propagation(X_train, W1, b1, W2, b2)
        custo_treinamento = calcular_custo(y_train, y_hat_train)
        custos_treinamento.append(custo_treinamento)

        _, y_hat_test = forward_propagation(X_test, W1, b1, W2, b2)
        custo_teste = calcular_custo(y_test, y_hat_test)
        custos_teste.append(custo_teste)

        print(f"Época {epoch+1}/{epochs}, Custo Treinamento: {custo_treinamento:.4f}, Custo Teste: {custo_teste:.4f}")

    return W1, b1, W2, b2, custos_treinamento, custos_teste

# Avaliação do modelo
def avaliar_modelo(X_test, y_test, W1, b1, W2, b2):
    _, y_hat_test = forward_propagation(X_test, W1, b1, W2, b2)
    y_pred = np.argmax(y_hat_test, axis=1)
    accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
    return accuracy

# Carregamento e pré-processamento dos dados
X_train, y_train, X_test, y_test = carregar_e_preprocessar_dados()

# Hiperparâmetros (você pode ajustar)
hidden_size = 128
learning_rate = 0.01
epochs = 50
batch_size = 64

# Treinamento do modelo
W1, b1, W2, b2, custos_treinamento, custos_teste = treinar_modelo(X_train, y_train, X_test, y_test, hidden_size, learning_rate, epochs, batch_size)

# Avaliação do modelo
accuracy = avaliar_modelo(X_test, y_test, W1, b1, W2, b2)
print(f"Acurácia nos dados de teste: {accuracy:.4f}")

# Plot dos custos
plt.plot(custos_treinamento, label="Treinamento")
plt.plot(custos_teste, label="Teste")
plt.xlabel("Épocas")
plt.ylabel("Custo")
plt.legend()
plt.show()