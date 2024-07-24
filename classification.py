import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# STAGE 1: Data Preparation

image_folder = "data/flying_vehicles"

images = []
labels = []

for class_label in range(4):
    class_folder = os.path.join(image_folder, str(class_label))

    for file_name in os.listdir(class_folder):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(class_folder, file_name)

            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
            img_array = tf.keras.preprocessing.image.img_to_array(img)

            images.append(img_array)
            labels.append(class_label)

X = np.array(images)
y = np.array(labels)

# STAGE 2: Data Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STAGE 3: Data Preprocessing

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# STAGE 4: Data Augmentation

image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
image_data_generator.fit(X_train)

# STAGE 5: Model Architecture

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),  # Input layer added
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation="softmax")  # 4 classes
])

# STAGE 6: Model Compilation

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# STAGE 7: Model Training

history = model.fit(
    image_data_generator.flow(X_train, y_train, batch_size=32),
    epochs=100,
    validation_data=(X_test, y_test)
)

# Save the trained model
model.save('flying_vehicles_model.h5')
print("Model saved as 'flying_vehicles_model.h5'")

# STAGE 8: Model Evaluation

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes))

# STAGE 9: Visualization

def make_accuracy_plot(history):
    import seaborn as sns
    sns.set()
    acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, acc, label="Training Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Validation Accuracy", marker="o")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

def make_loss_plot(history):
    import seaborn as sns
    sns.set()
    loss, val_loss = history.history["loss"], history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, loss, label="Training Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

make_accuracy_plot(history)
make_loss_plot(history)


