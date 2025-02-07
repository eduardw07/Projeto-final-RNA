import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

# Carregar o dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os dados (valores de pixel entre 0 e 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten (transformar 28x28 em um vetor de 784 dimensões)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Converter os rótulos para one-hot encoding
y_train_ohe = tf.keras.utils.to_categorical(y_train, 10)
y_test_ohe = tf.keras.utils.to_categorical(y_test, 10)

# Construção do modelo MLP
model = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),  # Primeira camada oculta
    Dropout(0.2),  # Regularização para evitar overfitting
    Dense(128, activation='relu'),  # Segunda camada oculta
    Dropout(0.2),
    Dense(10, activation='softmax')  # Camada de saída
])

# Compilar o modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(x_train, y_train_ohe, epochs=10, batch_size=32, validation_data=(x_test, y_test_ohe))

# Avaliação do modelo
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Mostrar matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - MLP com Keras')
plt.show()

# Exibir acurácia final
final_accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Acurácia Final: {final_accuracy * 100:.2f}%')


# Implementação do zero (sem TensorFlow/Keras)
class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = [
            np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / input_size),
            np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2.0 / hidden_sizes[0]),
            np.random.randn(hidden_sizes[1], output_size) * np.sqrt(2.0 / hidden_sizes[1])
        ]
        self.biases = [
            np.zeros((1, hidden_sizes[0])),
            np.zeros((1, hidden_sizes[1])),
            np.zeros((1, output_size))
        ]

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.z1 = np.dot(x, self.weights[0]) + self.biases[0]
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights[1]) + self.biases[1]
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.weights[2]) + self.biases[2]
        self.a3 = self.softmax(self.z3)
        return self.a3

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

    def backward(self, x, y_true):
        m = y_true.shape[0]
        dz3 = self.a3 - y_true
        dw3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = np.dot(dz3, self.weights[2].T) * (self.a2 > 0)
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.weights[1].T) * (self.a1 > 0)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.weights[2] -= self.learning_rate * dw3
        self.biases[2] -= self.learning_rate * db3
        self.weights[1] -= self.learning_rate * dw2
        self.biases[1] -= self.learning_rate * db2
        self.weights[0] -= self.learning_rate * dw1
        self.biases[0] -= self.learning_rate * db1


# Criando e treinando a rede MLP manualmente
mlp = MLP(input_size=784, hidden_sizes=[256, 128], output_size=10, learning_rate=0.01)
for epoch in range(10):
    predictions = mlp.forward(x_train)
    loss = mlp.compute_loss(y_train_ohe, predictions)
    mlp.backward(x_train, y_train_ohe)
    print(f'Época {epoch + 1}, Loss: {loss:.4f}')
