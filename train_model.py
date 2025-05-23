import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import preprocess_input
from pdf2image import convert_from_path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
import shutil
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from collections import defaultdict

K.clear_session()

IMAGE_SIZE = (512, 512)
MODEL_PATH = "resnet50_authenticity.h5"
TRAIN_DIR = 'data/train'
BATCH_SIZE = 16
EPOCHS = 20

def convert_pdf_to_images(pdf_path, output_folder, max_pages=2):
    pages = convert_from_path(pdf_path, 300)
    page_images = []
    for i, page in enumerate(pages[:max_pages]):  # Limit pages to avoid class bias
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        page.save(img_path, 'PNG')
        page_images.append(img_path)
    return page_images

def load_all_data(directory):
    file_paths = []
    labels = []

    for subdir, _, files in os.walk(directory):
        class_name = os.path.basename(subdir)
        if class_name == os.path.basename(directory):
            continue
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.lower().endswith('.pdf'):
                output_folder = os.path.join(subdir, 'temp_images')
                os.makedirs(output_folder, exist_ok=True)
                images = convert_pdf_to_images(file_path, output_folder)
                file_paths.extend(images)
                labels.extend([class_name] * len(images))
            elif file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(file_path)
                labels.append(class_name)

    images = []
    for file_path in file_paths:
        img = load_img(file_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        images.append(img_array)

    temp_dirs = [os.path.join(subdir, 'temp_images') for subdir in os.listdir(directory)]
    for td in temp_dirs:
        if os.path.exists(td):
            shutil.rmtree(td)

    return np.array(images), np.array(labels)

def train_model():
    images, labels = load_all_data(TRAIN_DIR)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_one_hot = to_categorical(labels_encoded)

    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
    )

    train_images = preprocess_input(train_images)
    val_images = preprocess_input(val_images)

    train_labels_one_hot = to_categorical(train_labels)
    val_labels_one_hot = to_categorical(val_labels)

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    base_model.trainable = False  

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(label_encoder.classes_), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')
    earlystop = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=1e-6)

    train_datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1,
                                       height_shift_range=0.1, horizontal_flip=True)
    val_datagen = ImageDataGenerator()

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights_array))

    model.fit(
        train_datagen.flow(train_images, train_labels_one_hot, batch_size=BATCH_SIZE),
        validation_data=val_datagen.flow(val_images, val_labels_one_hot, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=[checkpoint, earlystop, reduce_lr],
        class_weight=class_weights
    )

    for layer in base_model.layers[-30:]:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_datagen.flow(train_images, train_labels_one_hot, batch_size=BATCH_SIZE),
        validation_data=val_datagen.flow(val_images, val_labels_one_hot, batch_size=BATCH_SIZE),
        epochs=10,
        callbacks=[checkpoint, earlystop, reduce_lr],
        class_weight=class_weights
    )

    train_loss, train_acc = model.evaluate(train_datagen.flow(train_images, train_labels_one_hot))
    val_loss, val_acc = model.evaluate(val_datagen.flow(val_images, val_labels_one_hot))
    print(f"Train Accuracy: {train_acc:.2f}, Validation Accuracy: {val_acc:.2f}")

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    model.save(MODEL_PATH)
    return model, label_encoder

model, label_encoder = train_model()
