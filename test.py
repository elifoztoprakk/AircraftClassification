import os
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Veri Setlerini Eğitim ve Test Olarak Ayırma

def split_data(source_dir, train_dir, test_dir, test_size=0.2):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        train_category_path = os.path.join(train_dir, category)
        test_category_path = os.path.join(test_dir, category)
        
        if not os.path.exists(train_category_path):
            os.makedirs(train_category_path)
        if not os.path.exists(test_category_path):
            os.makedirs(test_category_path)
        
        images = os.listdir(category_path)
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
        
        for image in train_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(train_category_path, image))
        for image in test_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(test_category_path, image))

split_data('data/flying_vehicles', 'data/train', 'data/test')

#Veri Ön İşleme ve Artırma

# Veri artırma için ImageDataGenerator kullanımı
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def preprocess_and_augment_images(input_dir, output_dir, target_size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, target_size)
                img_array = np.expand_dims(img_resized, axis=0)
                
                # Veri artırma
                for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_category_path, save_prefix='aug', save_format='jpeg'):
                    break

preprocess_and_augment_images('data/train', 'processed_data/train')
preprocess_and_augment_images('data/test', 'processed_data/test')
#Modelin Tasarlanması ve Eğitilmesi

# Önceden eğitilmiş VGG16 modelini yükleyin, son katmanlar hariç
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Yeni sınıflandırma katmanları ekleyin
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

# Modeli oluşturun
model = Model(inputs=base_model.input, outputs=predictions)

# Öğrenme hızını zamanla azaltmak için öğrenme hızı programı
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = Adam(learning_rate=lr_schedule)

# Modeli derleyin
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Eğitim veri seti oluşturma
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'processed_data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

# Test veri seti oluşturma
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'processed_data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

# Modeli eğitme
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=test_generator
)
#Modelin Değerlendirilmesi ve Görselleştirme

# Modeli test veri seti üzerinde değerlendirme
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Tahminleri yapma
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print(classification_report(y_true, y_pred_classes, target_names=['Passenger Plane', 'Fighter Jet', 'Drone', 'Helicopter']))

# Eğitim ve doğrulama doğruluğunu görselleştirme
def make_accuracy_plot(history):
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

# Eğitim ve doğrulama kaybını görselleştirme
def make_loss_plot(history):
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
#Modelin Kaydedilmesi

model.save("aircraft_classification_model.h5")