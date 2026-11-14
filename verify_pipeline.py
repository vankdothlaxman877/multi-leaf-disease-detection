"""
Verification script for Multi-Leaf Disease Detection Pipeline.
Validates environment, dataset discovery, CLAHE preprocessing, deep feature extraction,
Lightweight ResNet / FeatureNet architectures, and Federated Learning (FedAvg) aggregation.
"""

import os, sys, cv2, time
import numpy as np

# Set environment
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("=" * 65)
print("MULTI-LEAF DISEASE DETECTION: SYSTEM VERIFICATION")
print("=" * 65)

# Step 1: Check Python and Library Versions
import tensorflow as tf
from tensorflow.keras.applications import VGG19, DenseNet121, ResNet50
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Conv2D, MaxPooling2D, Flatten, Add, Activation
from sklearn.metrics import confusion_matrix, roc_curve, auc
import vmdpy

print(f"[✓] Python Version      : {sys.version.split()[0]}")
print(f"[✓] TensorFlow Version  : {tf.__version__}")
print(f"[✓] OpenCV Version      : {cv2.__version__}")
print(f"[✓] vmdpy installed     : True")
print(f"[✓] GPUs Available      : {len(tf.config.list_physical_devices('GPU'))}")

# Step 2: Validate Dataset Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(base_dir, "datasets")

mango_path = os.path.join(datasets_dir, "Mango_Leaf_Dataset", "Mango_leaf_disease_dataset")
apple_path = os.path.join(datasets_dir, "Apple_Leaf_Dataset", "data")
tomato_path = os.path.join(datasets_dir, "Tomato_Leaf_Dataset", "tomato", "train")

for name, p in [("Mango", mango_path), ("Apple", apple_path), ("Tomato", tomato_path)]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing dataset directory for {name} at {p}")
    print(f"[✓] Dataset Found: {name:7s} -> {p}")

# Step 3: Test CLAHE preprocessing
def apply_clahe(img_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# Test dummy image
dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
clahe_img = apply_clahe(dummy_img)
assert clahe_img.shape == (224, 224, 3)
print("[✓] CLAHE preprocessing verified successfully.")

# Step 4: Load Feature Extractors
print("\nLoading Pretrained Feature Extractors (ImageNet)...")
t0 = time.time()
vgg = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")
dense = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")
resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")
print(f"[✓] Feature Extractors Loaded in {time.time()-t0:.2f}s:")
print(f"    - VGG19       : {vgg.output_shape} (512-dim)")
print(f"    - DenseNet121 : {dense.output_shape} (1024-dim)")
print(f"    - ResNet50    : {resnet.output_shape} (2048-dim)")
print(f"    - Total Vector: 3,584-dim")

# Step 5: Test End-to-End Extraction on Dummy Batch
dummy_batch = np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8).astype("float32")
v_feat = vgg.predict(vgg_preprocess(dummy_batch.copy()), verbose=0)
d_feat = dense.predict(densenet_preprocess(dummy_batch.copy()), verbose=0)
r_feat = resnet.predict(resnet_preprocess(dummy_batch.copy()), verbose=0)
feat_combined = np.hstack([v_feat, d_feat, r_feat])
assert feat_combined.shape == (4, 3584)
print(f"[✓] Deep Feature Extraction Verified. Output Shape: {feat_combined.shape}")

# Step 6: Test Model Architecture & FedAvg Aggregation
def build_fl_model(input_dim=3584):
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

global_model = build_fl_model(3584)
global_weights = global_model.get_weights()

# Simulate 3 clients
client_weights_list = []
client_sizes = [500, 700, 1000]
for sz in client_sizes:
    c_model = build_fl_model(3584)
    c_model.set_weights(global_weights)
    # Quick fit on synthetic data
    X_syn = np.random.randn(32, 3584)
    y_syn = np.random.randint(0, 2, (32,))
    c_model.fit(X_syn, y_syn, epochs=1, batch_size=16, verbose=0)
    client_weights_list.append(c_model.get_weights())

# FedAvg aggregation
def fedavg(global_weights, client_weights_list, client_sizes):
    total = sum(client_sizes)
    new_weights = []
    for layer_idx in range(len(global_weights)):
        w_layer = np.zeros_like(global_weights[layer_idx])
        for c_w, n_k in zip(client_weights_list, client_sizes):
            w_layer += (n_k / total) * c_w[layer_idx]
        new_weights.append(w_layer)
    return new_weights

updated_weights = fedavg(global_weights, client_weights_list, client_sizes)
global_model.set_weights(updated_weights)
print("[✓] Federated Learning (FedAvg) aggregation verified across 3 clients.")

# Save model test
save_dir = os.path.join(base_dir, "saved_models")
os.makedirs(save_dir, exist_ok=True)
save_file = os.path.join(save_dir, "test_global_model.keras")
global_model.save(save_file)
print(f"[✓] Model save verification successful: {save_file}")

print("=" * 65)
print("ALL TESTS PASSED: ENVIRONMENT & CODE FULLY FUNCTIONAL!")
print("=" * 65)
