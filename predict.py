import cv2
import numpy as np
import tensorflow as tf

# Загрузка обученной модели
model = tf.keras.models.load_model('models/emotion_model.h5')

# Список эмоций
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def recognize_emotion(image_path):
    # Загрузка изображения
    frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print("Ошибка: Не удалось загрузить изображение.")
        return

    # Изменение размера изображения до 48x48
    resized = cv2.resize(frame, (48, 48))
    normalized = resized.astype('float32') / 255.0
    input_image = np.expand_dims(normalized, axis=0)
    input_image = np.expand_dims(input_image, axis=-1)

    # Предсказание эмоции
    preds = model.predict(input_image)[0]
    emotion = EMOTIONS[preds.argmax()]

    # Вывод эмоции в консоль
    print(f"Распознанная эмоция: {emotion}")

if __name__ == "__main__":
    image_path = "imgs/fear.jpg"  # Укажите путь к вашему изображению
    recognize_emotion(image_path)