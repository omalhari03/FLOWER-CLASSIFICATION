if val_accuracy > 0.7:
    val_loss_ft, val_accuracy_ft = optimized_model.evaluate(val_ds_enhanced)
    print(f"Validation Accuracy after Fine-Tuning: {val_accuracy_ft:.4f}")

def plot_training_history(history, title='Training History'):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(optimized_history, 'Initial Training')

optimized_model.save('optimized_flowers_model.keras')
print("Optimized model saved successfully!")

print("=== COMPLEXITY AND EFFICIENCY ANALYSIS ===")

# 1. Model Size Comparison
original_params = 2686949  # From your original model
optimized_params = optimized_model.count_params()
reduction_percentage = ((original_params - optimized_params) / original_params) * 100

print(f"Original Model Parameters: {original_params:,}")
print(f"Optimized Model Parameters: {optimized_params:,}")
print(f"Parameter Reduction: {reduction_percentage:.2f}%")

# 2. Memory Usage Analysis
def get_model_memory_usage(model):
    """Calculate approximate memory usage of model"""
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    # Approximate memory: 4 bytes per parameter (float32)
    memory_mb = (total_params * 4) / (1024 * 1024)
    return memory_mb

original_memory = 10.25  # From your original model summary
optimized_memory = get_model_memory_usage(optimized_model)
memory_reduction = ((original_memory - optimized_memory) / original_memory) * 100

print(f"\nOriginal Model Memory: {original_memory:.2f} MB")
print(f"Optimized Model Memory: {optimized_memory:.2f} MB")
print(f"Memory Reduction: {memory_reduction:.2f}%")

# 3. Inference Speed Test
import time

def test_inference_speed(model, test_dataset, num_runs=10):
    """Test inference speed of the model"""
    times = []

    for images, _ in test_dataset.take(num_runs):
        start_time = time.time()
        _ = model.predict(images, verbose=0)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    return avg_time, fps

# Test inference speed
avg_time, fps = test_inference_speed(optimized_model, val_ds_enhanced)
print(f"\nAverage Inference Time per Batch: {avg_time:.4f} seconds")
print(f"Approximate FPS: {fps:.2f}")

# 4. Computational Complexity (FLOPs)
def estimate_flops(model):
    """Estimate FLOPs of the model"""
    try:
        from tensorflow.python.profiler import model_analyzer
        from tensorflow.python.profiler import option_builder

        profile = model_analyzer.profile(
            tf.compat.v1.get_default_graph(),
            options=option_builder.ProfileOptionBuilder.float_operation()
        )
        return profile.total_float_ops
    except:
        # Fallback estimation
        flops = optimized_model.count_params() * 2  # Rough estimate: 2 FLOPs per parameter
        return flops

estimated_flops = estimate_flops(optimized_model)
print(f"\nEstimated FLOPs: {estimated_flops:,}")

# 5. Accuracy vs Complexity Comparison
print(f"\n=== ACCURACY VS COMPLEXITY ===")
print(f"Model Parameters: {optimized_params:,}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Parameters per Accuracy Point: {optimized_params/val_accuracy:,.0f}")

# 6. Test with your own image
def predict_custom_image(model, image_path, class_names):
    """Predict class for a custom image"""
    # Load and preprocess image
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Get top 3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3_classes = [class_names[i] for i in top3_indices]
    top3_confidences = [predictions[0][i] for i in top3_indices]

    # Display results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Input Image\nPredicted: {class_names[predicted_class]}\nConfidence: {confidence:.3f}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top3_classes)))
    bars = plt.barh(range(len(top3_classes)), top3_confidences, color=colors)
    plt.xlabel('Confidence')
    plt.ylabel('Class')
    plt.title('Top 3 Predictions')
    plt.yticks(range(len(top3_classes)), top3_classes)

    # Add confidence values to bars
    for i, (bar, confidence) in enumerate(zip(bars, top3_confidences)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{confidence:.3f}', ha='left', va='center')

    plt.tight_layout()
    plt.show()

    return predicted_class, confidence, top3_classes, top3_confidences

# Test with a sample image from validation set first
print("\n=== TESTING WITH VALIDATION IMAGE ===")
for test_images, test_labels in val_ds_enhanced.take(1):
    # Take first image from batch
    test_image = test_images[0]
    true_label = test_labels[0].numpy()

    # Display true class
    print(f"True class: {class_names[true_label]}")

    # Make prediction
    img_array = tf.expand_dims(test_image, 0)
    predictions = optimized_model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    print(f"Predicted: {class_names[predicted_class]} with confidence {confidence:.3f}")

    # Display image with prediction
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title(f'True: {class_names[true_label]}\nPred: {class_names[predicted_class]}\nConf: {confidence:.3f}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # Show confidence distribution
    plt.barh(class_names, predictions[0])
    plt.xlabel('Confidence')
    plt.title('Class Confidence Distribution')
    plt.tight_layout()
    plt.show()

    break

# 7. Performance Metrics
print("\n=== PERFORMANCE METRICS ===")

# Calculate precision, recall, F1-score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Get all predictions and true labels
y_true = []
y_pred = []

for images, labels in val_ds_enhanced:
    predictions = optimized_model.predict(images, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(pred_classes)

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 8. Model Architecture Visualization
print("\n=== MODEL ARCHITECTURE ===")
try:
    tf.keras.utils.plot_model(
        optimized_model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )
    print("Model architecture diagram saved as 'model_architecture.png'")
except:
    print("Could not generate model architecture diagram")

# 9. Instructions for testing with your own image
print("\n" + "="*50)
print("TO TEST WITH YOUR OWN IMAGE:")
print("1. Upload your flower image to the notebook")
print("2. Run: predicted_class, confidence, top3, top3_conf = predict_custom_image(optimized_model, 'your_image.jpg', class_names)")
print("3. The function will display the image and predictions")
print("="*50)

# Save the analysis results
analysis_results = {
    'parameters': optimized_params,
    'parameter_reduction_percentage': reduction_percentage,
    'memory_mb': optimized_memory,
    'memory_reduction_percentage': memory_reduction,
    'inference_time': avg_time,
    'fps': fps,
    'estimated_flops': estimated_flops,
    'validation_accuracy': val_accuracy,
    'efficiency_ratio': optimized_params / val_accuracy
}

print(f"\nFinal Analysis Results:")
for key, value in analysis_results.items():
    if isinstance(value, float):
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    else:
        print(f"{key.replace('_', ' ').title()}: {value:,}")
