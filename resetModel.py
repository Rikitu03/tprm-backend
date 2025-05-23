from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as K
import pickle

# Define the number of classes your model is expected to classify
# 🔁 Change this according to your current use case
num_classes = 6 # e.g., 'BIR', 'GIS', 'FS'

def reset_model_and_encoder():
    """Reset the model and encoder to their initial states."""

    # Recreate model architecture with no pre-trained weights
    base_model = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    label_encoder = LabelEncoder()

    # Clear session
    K.clear_session()

    return model, label_encoder

model, label_encoder = reset_model_and_encoder()

# Save reset model and encoder
model.save('reset_resnet50_model.h5')
with open('reset_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Model and Label Encoder have been reset.")
