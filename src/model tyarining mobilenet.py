def create_enhanced_mobilenet_model(num_classes):
    """
    Enhanced MobileNetV3 model with improvements for better accuracy and efficiency
    """
    # Load pre-trained MobileNetV3Large with optimized settings
    base_model = MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        include_preprocessing=False,
        alpha=1.0,
        minimalistic=True
    )

    # Strategic fine-tuning - unfreeze later layers
    base_model.trainable = True
    # Freeze early layers, unfreeze later layers
    for layer in base_model.layers[:150]:
        layer.trainable = False

    # Enhanced architecture
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=True)

    # Use GlobalAveragePooling for better performance
    x = GlobalAveragePooling2D()(x)

    # Enhanced regularization strategy
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)

    # Additional dense layer with regularization
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    # Output layer - make sure this matches the number of classes
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Use sparse_categorical_crossentropy for tf.data format (integer labels)
    model.compile(
        optimizer=Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Recreate the model with the correct number of classes
enhanced_model = create_enhanced_mobilenet_model(num_classes)
enhanced_model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Stop training if val_loss doesn’t improve for 5 epochs
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# Reduce LR if val_loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# Save the best model during training
checkpoint = ModelCheckpoint(
    filepath="enhanced_mobilenetv3_best.h5",
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False
)

# Combine into a list
enhanced_callbacks = [early_stop, reduce_lr, checkpoint]

print("=== Testing Enhanced MobileNetV3 (2 epochs) ===")
test_history = simple_train_model(
    enhanced_model,
    train_ds,
    val_ds,
    enhanced_callbacks,
    epochs=2
)

print("=== Data Pipeline Verification ===")

# Check training data
train_count = 0
for images, labels in train_ds:
    print(f"Train batch {train_count}: images={images.shape}, labels={labels.shape}")
    print(f"Label range: {labels.numpy().min()} to {labels.numpy().max()}")
    train_count += 1
    if train_count >= 2:  # Just check first 2 batches
        break

# Check validation data
val_count = 0
for images, labels in val_ds:
    print(f"Val batch {val_count}: images={images.shape}, labels={labels.shape}")
    print(f"Label range: {labels.numpy().min()} to {labels.numpy().max()}")
    val_count += 1
    if val_count >= 2:  # Just check first 2 batches
        break

print(f"Number of training batches: {train_count}")
print(f"Number of validation batches: {val_count}")

print("=== Testing Single Prediction ===")

# Get one batch
for test_images, test_labels in train_ds.take(1):
    break

print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Try a prediction
try:
    predictions = enhanced_model.predict(test_images[:1])  # Just first image
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction: {predictions}")
    print(f"Predicted class: {np.argmax(predictions)}")
    print(f"True label: {test_labels[0].numpy()}")
except Exception as e:
    print(f"Prediction error: {e}")


def create_optimized_mobilenet_model(num_classes):
    base_model = MobileNetV3Large(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        include_preprocessing=False,
        alpha=0.75,
        minimalistic=False
    )
    base_model.trainable = False

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

optimized_model = create_optimized_mobilenet_model(num_classes)
optimized_model.summary()

def train_optimized_model(model, train_ds, val_ds, callbacks, class_weights, epochs=EPOCHS):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
    )
    return history

optimized_history = train_optimized_model(
    optimized_model,
    train_ds_enhanced,
    val_ds_enhanced,
    enhanced_callbacks,
    class_weights,
    epochs=30
)
def unfreeze_and_fine_tune(model, train_ds, val_ds, callbacks, class_weights, epochs=10):
    model.trainable = True
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR/10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
    )
    return fine_tune_history

if val_accuracy > 0.7:
    fine_tune_history = unfreeze_and_fine_tune(
        optimized_model,
        train_ds_enhanced,
        val_ds_enhanced,
        enhanced_callbacks,
        class_weights,
        epochs=15
    )
