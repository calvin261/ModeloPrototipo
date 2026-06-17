# Guía de Dataset LSEC

## Introducción

Esta guía describe cómo crear y preparar un dataset de Lenguaje de Señas Ecuatoriano (LSEC) para entrenar el modelo de reconocimiento.

## Requisitos del Dataset

### Cantidad de Datos

| Etapa | Señas | Muestras/Seña | Total Muestras |
|-------|-------|---------------|----------------|
| MVP | 50-100 | 50-100 | 5,000-10,000 |
| Producción | 500+ | 100-200 | 50,000-100,000 |

### Especificaciones de Video

- **Resolución**: 640x480 mínimo
- **FPS**: 30 FPS
- **Formato**: MP4 (H.264)
- **Duración**: 2-5 segundos por seña
- **Cámara**: Frontal (como en uso real)

### Condiciones de Grabación

1. **Iluminación**:
   - Natural (ventana)
   - Artificial (lámpara)
   - Mixta
   - Diferentes intensidades

2. **Fondos**:
   - Pared lisa
   - Oficina
   - Hogar
   - Exteriores (con precaución)

3. **Distancia**:
   - Cerca (50cm)
   - Media (1m)
   - Lejos (1.5m)

4. **Ángulos**:
   - Frontal
   - Ligeramente lateral (±15°)

5. **Participantes**:
   - Diversidad de edad
   - Diversidad de género
   - Diferentes tonos de piel
   - Diferentes velocidades de ejecución

## Vocabulario Prioritario

### Nivel 1: Alfabeto y Números (37 señas)

```
A, B, C, D, E, F, G, H, I, J, K, L, M, N, Ñ, O, P, Q, R, S, T, U, V, W, X, Y, Z
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

### Nivel 2: Saludos y Cortesía (15 señas)

| Seña | Ecuatorianismo |
|------|----------------|
| Hola | ¡Qué más! |
| Buenos días | Buenos días |
| Buenas tardes | Buenas tardes |
| Buenas noches | Buenas noches |
| Adiós | ¡Chao! |
| Gracias | Gracias, pana |
| Por favor | Por fa |
| Perdón | Disculpe |
| Mucho gusto | Encantado |
| ¿Cómo estás? | ¿Qué fue? |
| Bien | Chévere |
| Mal | Está feo |
| Sí | Sí, pues |
| No | Nel |
| Más o menos | Ahí nomás |

### Nivel 3: Familia (12 señas)

| Seña | Ecuatorianismo |
|------|----------------|
| Mamá | La vieja |
| Papá | El viejo |
| Hermano/a | Ñaño/a |
| Hijo/a | Guagua |
| Abuelo/a | Abuelito/a |
| Tío/a | Tío/a |
| Primo/a | Primo/a |
| Esposo/a | Mi marido/mujer |
| Amigo/a | Pana |
| Familia | La fami |
| Bebé | El guagua |
| Persona | Gente |

### Nivel 4: Preguntas (10 señas)

| Seña | Ecuatorianismo |
|------|----------------|
| ¿Qué? | ¿Qué cosa? |
| ¿Quién? | ¿Quién es? |
| ¿Dónde? | ¿Dónde es? |
| ¿Cuándo? | ¿Cuándo es? |
| ¿Cómo? | ¿Cómo así? |
| ¿Por qué? | ¿Por qué pues? |
| ¿Cuánto? | ¿Cuánto sale? |
| ¿Cuál? | ¿Cuál es? |
| ¿Hay? | ¿Hay o no hay? |
| ¿Puedes? | ¿Puedes o qué? |

### Nivel 5: Acciones Comunes (15 señas)

```
Comer, Beber, Dormir, Trabajar, Estudiar, Ir, Venir, 
Querer, Poder, Necesitar, Ayudar, Entender, Esperar,
Comprar, Pagar
```

### Nivel 6: Emergencias (10 señas)

```
Ayuda, Doctor, Hospital, Policía, Peligro, 
Urgente, Dolor, Enfermo, Accidente, Llamar
```

## Proceso de Recolección

### Paso 1: Contactar Signantes

1. **FENASEC** (Federación Nacional de Sordos del Ecuador)
   - Web: http://www.fenasec.org.ec
   - Solicitar colaboración oficial
   - Obtener contacto de signantes nativos

2. **Asociaciones Regionales**:
   - Quito: Asociación de Sordos de Pichincha
   - Guayaquil: Asociación de Sordos del Guayas
   - Cuenca: Asociación de Sordos del Azuay

3. **Instituciones Educativas**:
   - Escuelas para sordos
   - Programas de interpretación

### Paso 2: Preparar Sesión de Grabación

```
Materiales necesarios:
□ Cámara/smartphone con soporte
□ Iluminación adicional
□ Fondo neutro (tela o cartón)
□ Lista de señas a grabar
□ Formulario de consentimiento
□ Compensación para participantes
```

### Paso 3: Protocolo de Grabación

1. **Inicio de sesión**:
   - Explicar el proyecto
   - Firmar consentimiento informado
   - Configurar cámara

2. **Por cada seña**:
   ```
   a) Mostrar tarjeta con palabra/concepto
   b) Signante practica (no grabar)
   c) Grabar 3-5 repeticiones
   d) Verificar calidad
   e) Siguiente seña
   ```

3. **Variaciones a capturar**:
   - Velocidad normal
   - Velocidad lenta
   - Mano dominante derecha
   - Mano dominante izquierda (si aplica)

### Paso 4: Anotación de Datos

#### Formato de Archivo

```
dataset/
├── raw_videos/
│   ├── signer_001/
│   │   ├── hola_001.mp4
│   │   ├── hola_002.mp4
│   │   └── ...
│   └── signer_002/
│       └── ...
├── annotations/
│   └── annotations.json
└── landmarks/
    └── extracted/
```

#### Esquema de Anotación (JSON)

```json
{
  "videos": [
    {
      "id": "hola_001",
      "file": "raw_videos/signer_001/hola_001.mp4",
      "signer_id": "signer_001",
      "sign": {
        "name": "Hola",
        "spanish": "Hola",
        "ecuadorian": "¡Qué más!",
        "category": "GREETINGS",
        "is_dynamic": true
      },
      "metadata": {
        "duration_ms": 2500,
        "fps": 30,
        "resolution": "640x480",
        "lighting": "natural",
        "background": "plain_wall",
        "dominant_hand": "right"
      },
      "quality": {
        "hands_visible": true,
        "face_visible": true,
        "blur_level": "low",
        "approved": true
      }
    }
  ]
}
```

## Extracción de Landmarks

### Script de Procesamiento

```python
import cv2
import mediapipe as mp
import numpy as np
import json

mp_hands = mp.solutions.hands

def extract_landmarks(video_path):
    """
    Extrae landmarks de MediaPipe Hands de un video.
    
    Returns:
        List[np.ndarray]: Lista de arrays (21, 3) por frame
    """
    landmarks_sequence = []
    
    cap = cv2.VideoCapture(video_path)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir BGR a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar frame
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Tomar primera mano detectada
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extraer 21 landmarks x 3 coordenadas
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in hand_landmarks.landmark
                ])
                
                landmarks_sequence.append(landmarks.flatten())
            else:
                # Padding con ceros si no hay manos
                landmarks_sequence.append(np.zeros(63))
    
    cap.release()
    
    return np.array(landmarks_sequence)

def process_dataset(annotations_path, output_dir):
    """
    Procesa todo el dataset extrayendo landmarks.
    """
    with open(annotations_path) as f:
        annotations = json.load(f)
    
    for video in annotations['videos']:
        video_path = video['file']
        landmarks = extract_landmarks(video_path)
        
        # Guardar landmarks
        output_path = f"{output_dir}/{video['id']}.npy"
        np.save(output_path, landmarks)
        print(f"Processed: {video['id']} - Shape: {landmarks.shape}")
```

## Data Augmentation

### Técnicas Recomendadas

```python
def augment_landmarks(landmarks):
    """
    Aplica data augmentation a secuencias de landmarks.
    """
    augmented = []
    
    # 1. Original
    augmented.append(landmarks)
    
    # 2. Ruido gaussiano
    noise = np.random.normal(0, 0.01, landmarks.shape)
    augmented.append(landmarks + noise)
    
    # 3. Escalado
    scale = np.random.uniform(0.9, 1.1)
    augmented.append(landmarks * scale)
    
    # 4. Time warping (cambiar velocidad)
    stretched = time_warp(landmarks, factor=1.2)
    augmented.append(stretched)
    
    compressed = time_warp(landmarks, factor=0.8)
    augmented.append(compressed)
    
    # 5. Reflexión horizontal (espejo)
    mirrored = mirror_landmarks(landmarks)
    augmented.append(mirrored)
    
    return augmented

def time_warp(landmarks, factor):
    """Cambia la velocidad temporal de la secuencia."""
    n_frames = int(len(landmarks) * factor)
    indices = np.linspace(0, len(landmarks)-1, n_frames)
    return np.array([landmarks[int(i)] for i in indices])

def mirror_landmarks(landmarks):
    """Refleja horizontalmente (para simular mano izquierda)."""
    mirrored = landmarks.copy()
    # Invertir coordenada X
    mirrored[:, 0::3] = 1 - mirrored[:, 0::3]
    return mirrored
```

## División del Dataset

```python
from sklearn.model_selection import train_test_split

# Separar por signantes (no por muestras)
signer_ids = list(set([v['signer_id'] for v in annotations['videos']]))

train_signers, test_signers = train_test_split(
    signer_ids, 
    test_size=0.2, 
    random_state=42
)

train_signers, val_signers = train_test_split(
    train_signers, 
    test_size=0.15, 
    random_state=42
)

# Esto asegura que el modelo generalice a nuevos signantes
```

## Consentimiento Informado

### Plantilla

```
CONSENTIMIENTO INFORMADO PARA PARTICIPACIÓN EN DATASET LSEC

Proyecto: LSEC Translator - Aplicación de Traducción de Lengua de Señas Ecuatoriana
Investigador Principal: [Nombre]
Institución: [Universidad/Organización]

Propósito del Estudio:
Desarrollar una aplicación móvil que reconozca el Lenguaje de Señas Ecuatoriano (LSEC)
para facilitar la comunicación de personas sordas.

Participación:
- Grabaré videos de mis manos realizando señas LSEC
- Los videos serán usados para entrenar un modelo de inteligencia artificial
- Mi rostro puede aparecer parcialmente en las grabaciones

Privacidad:
- Los videos serán almacenados de forma segura
- No se compartirán con terceros sin permiso
- Puedo solicitar la eliminación de mis datos en cualquier momento

Derechos:
- Mi participación es voluntaria
- Puedo retirarme en cualquier momento
- Recibiré una copia de los resultados del proyecto

Compensación: [Especificar si aplica]

Firma: __________________ Fecha: __________
Nombre: __________________
CI: __________________
```

## Consideraciones Éticas

1. **Representatividad**: Incluir signantes de diferentes regiones del Ecuador
2. **Compensación justa**: Remunerar el tiempo de los participantes
3. **Validación comunitaria**: Verificar señas con expertos en LSEC
4. **Accesibilidad**: Compartir el dataset con la comunidad (con consentimiento)
5. **Mejora continua**: Actualizar el dataset con feedback de usuarios

## Recursos Adicionales

### Contactos Útiles

- **FENASEC**: Federación Nacional de Sordos del Ecuador
- **CONADIS**: Consejo Nacional para la Igualdad de Discapacidades
- **Universidades**: Programas de lingüística de señas

### Referencias

1. LSEC Dictionary (if available)
2. Similar projects: ASL datasets (WLASL, MSASL)
3. MediaPipe Hands documentation
4. TensorFlow data pipeline guides
