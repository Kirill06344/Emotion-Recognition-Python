from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from utils.data_processing import load_and_preprocess_data
import matplotlib.pyplot as plt

TRAIN_DIR = 'data/facial-expression-dataset/train/train/'
TEST_DIR = 'data/facial-expression-dataset/test/test/'

input_shape = (48, 48, 1)  # Input image shape (height, width, channels)
output_class = 7  # Number of output classes (e.g., emotions)

# Loading and preprocess data
x_train, x_test, y_train, y_test = load_and_preprocess_data(TRAIN_DIR, TEST_DIR, output_class)

# Model creation
model = Sequential()

# Convolutional layers
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))  # First convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling to reduce spatial dimensions
model.add(Dropout(0.4))  # Dropout for regularization

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))  # Second convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))  # Third convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))  # Fourth convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())  # Flatten the feature maps into a 1D vector

# Fully connected layers
model.add(Dense(512, activation='relu'))  # First dense layer
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))  # Second dense layer
model.add(Dropout(0.3))

# Output layer
model.add(Dense(output_class, activation='softmax'))  # Output layer with softmax activation

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=128,  # Batch size for training
    epochs=100,  # Maximum number of epochs
    validation_data=(x_test, y_test),  # Validation dataset
    callbacks=[early_stopping],  # Callbacks for early stopping
)

# Save the trained model
model.save('models/emotion_model')

# Plot training results
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plots to a file
plt.tight_layout()
plt.savefig('results/model.png')  # Save the graphs to a file
plt.close()