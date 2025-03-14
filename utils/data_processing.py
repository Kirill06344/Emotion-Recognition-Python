import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path):
    # Загрузка данных
    data = pd.read_csv(csv_path)

    # Разделение данных на пиксели и метки
    pixels = data['pixels'].tolist()
    labels = data['emotion'].values

    # Преобразование строк пикселей в массивы NumPy
    images = []
    for pixel_sequence in pixels:
        image = np.array(pixel_sequence.split(), dtype='uint8').reshape(48, 48)
        images.append(image)

    images = np.array(images)
    labels = np.array(labels)

    # Нормализация изображений (значения пикселей приводятся к диапазону [0, 1])
    images = images.astype('float32') / 255.0

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # Добавление канального измерения (для CNN)
    X_train = np.expand_dims(X_train, axis=-1)  # Размерность: (num_samples, 48, 48, 1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test, y_train, y_test