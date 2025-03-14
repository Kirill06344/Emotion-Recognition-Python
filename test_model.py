import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# Загрузка обученной модели
model = tf.keras.models.load_model('models/emotion_model.h5')

# Список эмоций
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Функция для загрузки тестовых данных
def load_test_data(csv_path):
    # Загрузка данных
    data = pd.read_csv(csv_path)

    # Разделение на тестовые данные
    test_data = data[data['Usage'] == 'PublicTest']
    pixels = test_data['pixels'].tolist()
    labels = test_data['emotion'].values

    # Преобразование строк пикселей в массивы NumPy
    images = []
    for pixel_sequence in pixels:
        image = np.array(pixel_sequence.split(), dtype='uint8').reshape(48, 48)
        images.append(image)

    images = np.array(images).astype('float32') / 255.0
    labels = np.array(labels)

    # Добавление канального измерения
    images = np.expand_dims(images, axis=-1)

    return images, labels

# Функция для тестирования модели
def test_model(model, test_images, test_labels):
    # Предсказания модели
    y_pred_probs = model.predict(test_images)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Подсчёт общего количества и правильно предсказанных
    total_count = len(test_labels)
    correct_count = np.sum(y_pred == test_labels)
    accuracy = correct_count / total_count

    # Матрица ошибок
    cm = confusion_matrix(test_labels, y_pred)

    # Визуализация матрицы ошибок
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Сохранение графика в файл
    plt.savefig('results/confusion_matrix.png')  # Сохраняем в файл
    plt.close()  # Закрываем график, чтобы избежать предупреждений

    # Отчёт о классификации
    report = classification_report(test_labels, y_pred, target_names=EMOTIONS)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    # Загрузка тестовых данных
    test_images, test_labels = load_test_data('data/fer2013.csv')

    # Тестирование модели
    test_model(model, test_images, test_labels)