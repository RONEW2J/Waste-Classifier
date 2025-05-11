import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow messages
import tensorflow as tf
from tensorflow.keras import layers, callbacks, applications
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = os.path.join('data', 'dataset-resized')

TRASHNET_DIR = os.path.join('data', 'dataset-resized')
CUSTOM_DIR = os.path.join('data', 'custom_data')


def load_combined_dataset():
    """Load both TrashNet and custom datasets"""
    # Load TrashNet data
    trashnet_train = tf.keras.preprocessing.image_dataset_from_directory(
        TRASHNET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int')

    trashnet_val = tf.keras.preprocessing.image_dataset_from_directory(
        TRASHNET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int')

    # Load custom data if exists
    if os.path.exists(CUSTOM_DIR):
        custom_train = tf.keras.preprocessing.image_dataset_from_directory(
            CUSTOM_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='int')

        custom_val = tf.keras.preprocessing.image_dataset_from_directory(
            CUSTOM_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='int')

        # Combine datasets
        train_ds = trashnet_train.concatenate(custom_train)
        val_ds = trashnet_val.concatenate(custom_val)
        print("Using combined TrashNet + custom dataset")
    else:
        train_ds = trashnet_train
        val_ds = trashnet_val
        print("Using only TrashNet dataset")

    print(f"Training samples: {len(train_ds) * BATCH_SIZE}")
    print(f"Validation samples: {len(val_ds) * BATCH_SIZE}")
    return train_ds, val_ds, trashnet_train.class_names


def build_model():
    """Create model with transfer learning"""
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
    ])

    # Base model
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    # Model architecture
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Modern loss function
        metrics=['accuracy'])

    return model


def save_results(history, model, val_ds):
    """Save training results"""
    # Save training history
    history_dict = {
        'accuracy': [float(acc) for acc in history.history['accuracy']],
        'val_accuracy': [float(acc) for acc in history.history['val_accuracy']],
        'loss': [float(loss) for loss in history.history['loss']],
        'val_loss': [float(loss) for loss in history.history['val_loss']],
    }

    # Plot training metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['accuracy'], label='Train Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig('training_metrics.png')
    plt.close()

    # Confusion Matrix
    y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
    y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=val_ds.class_names)
    disp.plot(cmap='Blues')
    plt.xticks(rotation=45)
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()


def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)

    # Load data
    train_ds, val_ds, class_names = load_combined_dataset()
    print(f"Training samples: {len(train_ds) * BATCH_SIZE}")
    print(f"Validation samples: {len(val_ds) * BATCH_SIZE}")

    # Build model
    model = build_model()
    model.summary()

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            os.path.join('models', 'best_model.h5'),
            save_best_only=True,
            save_format='h5')
    ]

    # Train model
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1)

    # Save final model
    model.save(os.path.join('models', 'waste_classifier.h5'))
    print("Model saved successfully.")

    # Save results
    save_results(history, model, val_ds)
    print("Training results saved.")


if __name__ == '__main__':
    # Explicitly disable GPU if not available
    tf.config.set_visible_devices([], 'GPU')
    main()