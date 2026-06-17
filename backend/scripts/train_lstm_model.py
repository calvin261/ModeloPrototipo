"""
Script de entrenamiento del modelo LSTM para clasificación de señas LSEC.

Este script entrena un modelo Bi-LSTM cuantizado para reconocimiento
de Lenguaje de Señas Ecuatoriano (LSEC) usando secuencias de landmarks.

Arquitectura:
- Input: Secuencias de 30 frames x 63 features (21 landmarks x 3 coords)
- 2 capas Bi-LSTM de 128 unidades cada una
- Dropout de 0.3 para regularización
- Dense layer de 256 unidades
- Output: Softmax sobre N clases de señas

Target: Modelo TFLite cuantizado INT8 para inferencia móvil <30ms.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os

# Configuración
SEQUENCE_LENGTH = 30  # Frames por secuencia
FEATURES_PER_FRAME = 63  # 21 landmarks x 3 coordenadas
LSTM_UNITS = 128
DROPOUT_RATE = 0.3
DENSE_UNITS = 256
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Directorio de salida
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_lstm_model(num_classes: int) -> keras.Model:
    """
    Crea el modelo Bi-LSTM para clasificación de señas LSEC.
    
    Args:
        num_classes: Número de clases de señas a clasificar.
        
    Returns:
        Modelo Keras compilado.
    """
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME))
    
    # Normalización de entrada
    x = layers.BatchNormalization()(inputs)
    
    # Primera capa Bi-LSTM
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True)
    )(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Segunda capa Bi-LSTM
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=False)
    )(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Capas densas
    x = layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    # Capa de salida
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="LSEC_LSTM_Classifier")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_transformer_model(num_classes: int) -> keras.Model:
    """
    Modelo alternativo basado en Transformer ligero.
    
    Args:
        num_classes: Número de clases de señas.
        
    Returns:
        Modelo Keras compilado.
    """
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME))
    
    # Embedding posicional
    positions = tf.range(start=0, limit=SEQUENCE_LENGTH, delta=1)
    pos_embedding = layers.Embedding(input_dim=SEQUENCE_LENGTH, output_dim=FEATURES_PER_FRAME)(positions)
    x = inputs + pos_embedding
    
    # Transformer encoder block
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=32
    )(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    
    # Feed-forward network
    ff = layers.Dense(128, activation='relu')(x)
    ff = layers.Dense(FEATURES_PER_FRAME)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Clasificador
    x = layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="LSEC_Transformer_Classifier")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extrae características adicionales de los landmarks.
    
    Incluye:
    - Coordenadas relativas al centro de la mano
    - Velocidad entre frames consecutivos
    - Orientación de los dedos
    
    Args:
        landmarks: Array de forma (batch, sequence, 63)
        
    Returns:
        Array de características procesadas
    """
    # Por ahora, usar landmarks raw
    # TODO: Agregar features de velocidad y orientación
    return landmarks


def augment_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Aplica data augmentation para mejorar generalización.
    
    Técnicas:
    - Ruido gaussiano
    - Time warping
    - Escalado aleatorio
    - Rotación 2D
    
    Args:
        X: Datos de entrada (batch, sequence, features)
        y: Etiquetas
        
    Returns:
        Datos aumentados
    """
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # Original
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Ruido gaussiano
        noise = np.random.normal(0, 0.01, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])
        
        # Escalado aleatorio
        scale = np.random.uniform(0.9, 1.1)
        augmented_X.append(X[i] * scale)
        augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)


def generate_synthetic_data(num_samples: int, num_classes: int) -> tuple:
    """
    Genera datos sintéticos para pruebas iniciales.
    
    NOTA: Reemplazar con dataset real de LSEC para entrenamiento de producción.
    
    Args:
        num_samples: Número de muestras a generar
        num_classes: Número de clases
        
    Returns:
        Tuple (X, y) con datos sintéticos
    """
    X = np.random.randn(num_samples, SEQUENCE_LENGTH, FEATURES_PER_FRAME).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    
    # Agregar patrones distinguibles por clase
    for i in range(num_samples):
        class_idx = y[i]
        # Agregar bias basado en clase para que el modelo pueda aprender
        X[i] += class_idx * 0.1
        X[i, :, class_idx % FEATURES_PER_FRAME] += 0.5
    
    return X, y


def convert_to_tflite(model: keras.Model, output_path: str, quantize: bool = True):
    """
    Convierte el modelo Keras a TensorFlow Lite con cuantización INT8.
    
    Args:
        model: Modelo Keras entrenado
        output_path: Ruta de salida para el archivo .tflite
        quantize: Si aplicar cuantización INT8
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        
        # Datos representativos para calibración
        def representative_dataset():
            for _ in range(100):
                data = np.random.randn(1, SEQUENCE_LENGTH, FEATURES_PER_FRAME).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32  # Mantener float32 en entrada
        converter.inference_output_type = tf.float32  # Mantener float32 en salida
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Modelo TFLite guardado en: {output_path}")
    print(f"Tamaño: {len(tflite_model) / 1024:.2f} KB")


def save_labels(labels: list, output_path: str):
    """
    Guarda las etiquetas de clase en formato JSON.
    
    Args:
        labels: Lista de etiquetas
        output_path: Ruta de salida
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def main():
    """
    Función principal de entrenamiento.
    """
    print("=" * 60)
    print("LSEC LSTM Model Training")
    print("=" * 60)
    
    # Definir clases de señas LSEC
    lsec_labels = [
        {"index": 0, "name": "Hola", "ecuadorian": "¡Qué más!"},
        {"index": 1, "name": "Gracias", "ecuadorian": "Gracias, pana"},
        {"index": 2, "name": "Por favor", "ecuadorian": "Por fa"},
        {"index": 3, "name": "Sí", "ecuadorian": "Sí, pues"},
        {"index": 4, "name": "No", "ecuadorian": "Nel"},
        {"index": 5, "name": "Bien", "ecuadorian": "Chévere"},
        {"index": 6, "name": "Mal", "ecuadorian": "Está feo"},
        {"index": 7, "name": "Amigo", "ecuadorian": "Pana"},
        {"index": 8, "name": "Familia", "ecuadorian": "La fami"},
        {"index": 9, "name": "Ayuda", "ecuadorian": "Dame una mano"},
        # Agregar más señas aquí...
    ]
    
    num_classes = len(lsec_labels)
    print(f"\nClases de señas: {num_classes}")
    
    # Generar datos sintéticos (reemplazar con dataset real)
    print("\nGenerando datos de entrenamiento...")
    X_train, y_train = generate_synthetic_data(1000, num_classes)
    X_val, y_val = generate_synthetic_data(200, num_classes)
    
    # Aplicar data augmentation
    print("Aplicando data augmentation...")
    X_train, y_train = augment_data(X_train, y_train)
    
    print(f"Datos de entrenamiento: {X_train.shape}")
    print(f"Datos de validación: {X_val.shape}")
    
    # Crear modelo
    print("\nCreando modelo Bi-LSTM...")
    model = create_lstm_model(num_classes)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'lsec_lstm_best.keras'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Entrenar
    print("\nIniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluar
    print("\nEvaluación final:")
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Loss: {test_loss:.4f}")
    
    # Guardar modelo Keras
    model.save(os.path.join(OUTPUT_DIR, 'lsec_lstm_classifier.keras'))
    
    # Convertir a TFLite
    print("\nConvirtiendo a TensorFlow Lite...")
    convert_to_tflite(
        model,
        os.path.join(OUTPUT_DIR, 'lsec_lstm_classifier.tflite'),
        quantize=True
    )
    
    # Guardar etiquetas
    save_labels(lsec_labels, os.path.join(OUTPUT_DIR, 'lsec_labels.json'))
    
    print("\n" + "=" * 60)
    print("Entrenamiento completado!")
    print("=" * 60)
    print(f"\nArchivos generados en '{OUTPUT_DIR}/':")
    print("- lsec_lstm_classifier.keras (modelo Keras)")
    print("- lsec_lstm_classifier.tflite (modelo cuantizado)")
    print("- lsec_labels.json (etiquetas de clases)")


if __name__ == "__main__":
    main()
