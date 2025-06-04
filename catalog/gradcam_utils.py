import numpy as np
import tensorflow as tf
import cv2
import os
import uuid

def generate_gradcam(model, img_path, predicted_class, layer_name=None, save_dir="static/results"):
    # Load image and preprocess
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Get model's last conv layer if not specified
    if not layer_name:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)[0]  # shape: (H, W, C)
    conv_outputs = conv_outputs[0]

    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Multiply each channel by the corresponding gradient
    conv_outputs = conv_outputs.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Average channels to get heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10  # normalize

    # Load original image (for overlay size)
    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))

    # Resize and colorize heatmap
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original, 0.5, heatmap_color, 0.5, 0)

    # Save result
    os.makedirs(save_dir, exist_ok=True)
    filename = f"gradcam_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(save_dir, filename)
    cv2.imwrite(path, overlay)

    return path
