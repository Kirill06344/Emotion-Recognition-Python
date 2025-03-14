import numpy as np
import tensorflow as tf
from keras.src.layers import BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping

from utils.data_processing import load_and_preprocess_data
import matplotlib.pyplot as plt

# Загрузка и предобработка данных
X_train, X_test, y_train, y_test = load_and_preprocess_data('data/fer2013.csv')

# Создание модели
model = Sequential()

# Первый сверточный блок
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Второй сверточный блок
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Полносвязные слои
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(7, activation='softmax'))  # 7 классов эмоций

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    validation_data=(X_test, y_test),
    epochs=30,
    callbacks=[early_stopping],
)

# Сохранение модели
model.save('models/emotion_model.h5')

# Построение графиков обучения
plt.figure(figsize=(12, 5))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Сохранение графиков
plt.tight_layout()
plt.savefig('results/model.png')  # Сохраняем график в файл
plt.close()