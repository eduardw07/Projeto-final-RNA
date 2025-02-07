import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
