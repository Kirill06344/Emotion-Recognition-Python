import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from utils.data_processing import load_dataset, extract_features

TRAIN_DIR = 'data/facial-expression-dataset/train/train/'
TEST_DIR = 'data/facial-expression-dataset/test/test/'

# Загрузка обученной модели
model = tf.keras.models.load_model('models/emotion_model.h5')

# Список эмоций
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Функция для тестирования модели
def test_model(model, test_images, test_labels):
    # Предсказания модели
    y_pred_probs = model.predict(test_images)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Преобразование one-hot encoded меток в одномерный массив индексов
    y_true = np.argmax(test_labels, axis=1)

    # Подсчёт общего количества и правильно предсказанных
    total_count = len(y_true)
    correct_count = np.sum(y_pred == y_true)
    accuracy = correct_count / total_count

    print(f"Accuracy: {accuracy:.4f}")

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)

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
    report = classification_report(y_true, y_pred, target_names=EMOTIONS)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    # Загрузка данных
    train_images, train_labels = load_dataset(TRAIN_DIR)
    test_images, test_labels = load_dataset(TEST_DIR)

    # Объединение данных
    all_images = train_images + test_images
    all_labels = train_labels + test_labels

    # Извлечение признаков
    all_features = extract_features(all_images)

    # Нормализация
    all_features = all_features / 255.0

    # Кодирование меток
    le = LabelEncoder()
    le.fit(all_labels)
    all_labels_encoded = le.transform(all_labels)

    # Преобразование меток в one-hot encoding
    all_labels_onehot = tf.keras.utils.to_categorical(all_labels_encoded, num_classes=7)

    # Тестирование модели
    test_model(model, all_features, all_labels_onehot)