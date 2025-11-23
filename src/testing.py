import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Set your image size (should match the model input)
IMG_SIZE = (224, 224)

# Load your saved optimized model
optimized_model = tf.keras.models.load_model('optimized_flowers_model.keras')
print("Model loaded successfully!")

# Define your flower classes
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # Replace with your classes

def predict_image_from_path(model, class_names, image_path):
    """Predict flower class for a given image path"""
    # Load and preprocess the image
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, IMG_SIZE)
        img_array = img_resized.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"OpenCV failed, using PIL: {e}")
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Top 3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    top3_classes = [class_names[i] for i in top3_indices]
    top3_confidences = [predictions[0][i] for i in top3_indices]

    # Display image and predictions
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.3f}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, len(top3_classes)))
    bars = plt.barh(range(len(top3_classes)), top3_confidences, color=colors)
    plt.xlabel('Confidence Score')
    plt.ylabel('Flower Class')
    plt.title('Top 3 Predictions')
    plt.yticks(range(len(top3_classes)), top3_classes)
    plt.xlim(0, 1)

    for i, (bar, conf) in enumerate(zip(bars, top3_confidences)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{conf:.3f}', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Print results
    print(f"\n{'='*50}")
    print("PREDICTION RESULTS:")
    print(f"{'='*50}")
    print(f"Most likely: {class_names[predicted_class]} (Confidence: {confidence:.3f})")
    print("\nTop 3 predictions:")
    for i, (cls, conf) in enumerate(zip(top3_classes, top3_confidences)):
        print(f"{i+1}. {cls}: {conf:.3f}")

    print("\nFull confidence distribution:")
    for i, (cls, conf) in enumerate(zip(class_names, predictions[0])):
        print(f"{cls}: {conf:.4f}")

    return predicted_class, confidence, top3_classes, top3_confidences

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Replace this path with your image path
    image_path = r"C:\Users\shari\Downloads\102501987_3cdb8e5394_n.jpg"
    predict_image_from_path(optimized_model, class_names, image_path)
