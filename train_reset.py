from tensorflow.keras import backend as K
import os

K.clear_session()

if os.path.exists("resnet50_authenticity.h5"):
    os.remove("resnet50_authenticity.h5")
if os.path.exists("label_encoder.pkl"):
    os.remove("label_encoder.pkl")
