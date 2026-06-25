# Plan de Implementación: Pipeline de ML Holístico para Señas Dinámicas (LSEC)

Este plan describe la reestructuración del pipeline de visión y aprendizaje automático para incorporar el seguimiento de manos, brazos y cara, resolviendo el problema de las señas dinámicas que dependen del contexto corporal.

---

## Opciones Disponibles para Revisión del Usuario

Para abordar el problema, te presento dos opciones con diferentes niveles de complejidad y cobertura técnica:

### Opción A (Recomendada - Solución Completa Holística)
Migración a un modelo de detección combinada de **Dos Manos + Pose Superior (Hombros, Codos, Muñecas)**.
* **Cómo funciona**: En Android se ejecutan en paralelo `HandLandmarker` (configurado para 2 manos) y `PoseLandmarker`. El origen del espacio de coordenadas pasa a ser el centro entre los hombros (o la nariz), manteniendo la relación de distancia entre las manos y el cuerpo.
* **Entrenamiento**: Se re-extrae todo el dataset LSC50 desde los videos crudos en `backend/data/public/LSC50` usando el nuevo extractor holístico de Python. Se entrena el `SignTransformer` con la nueva dimensión de entrada expandida (por ejemplo, 150 valores por frame en lugar de 63).
* **Pros**: Soluciona al 100% la confusión entre señas que tienen la misma forma de mano pero distinta ubicación corporal (frente vs. pecho), y añade soporte para dos manos y brazos.
* **Cons**: Alta complejidad técnica. Requiere re-entrenar el modelo en tu máquina (tardará aprox. 30-40 minutos).

### Opción B (Solución Parcial - Sólo Bimanual)
Ampliación del pipeline para detectar **Dos Manos** sin información de pose/cara.
* **Cómo funciona**: Se hace que `HandDetector` y el extractor de Python admitan `num_hands = 2`. La segunda mano se normaliza respecto a la muñeca de la primera mano para conservar la distancia relativa.
* **Entrenamiento**: Requiere re-extracción de landmarks y re-entrenamiento del clasificador con 126 valores de entrada (21 landmarks x 3 coords x 2 manos).
* **Pros**: Complejidad media. Soluciona el problema de señas que usan ambas manos (como "gracias" o "ayuda").
* **Cons**: No soluciona el problema si la seña dinámica depende de tocarse la frente, los hombros o el pecho (ambigüedad espacial persistente).

---

## Preguntas Abiertas para el Usuario

> [!IMPORTANT]
> Por favor, responde a estas preguntas antes de proceder:
> 1. ¿Cuál de las dos opciones prefieres implementar (**Opción A** u **Opción B**)?
> 2. ¿Tu computadora cuenta con recursos suficientes (GPU o CPU moderna) para correr el script de entrenamiento de Python que tardará de 20 a 40 minutos?
> 3. ¿Deseas que prepare los scripts para que puedas ejecutar el entrenamiento localmente tú mismo, o prefieres que yo intente lanzar el entrenamiento en segundo plano a través de la consola si tu entorno de desarrollo Python ya está configurado?

---

## Cambios Propuestos (Para Opción A - Holística)

### Componente de Visión (Android Frontend)

#### [MODIFY] [HandDetector.kt](file:///c:/Users/jeffe/Documents/Proyectos/ModeloPrototipo/frontend/app/src/main/java/com/lsec/translator/ml/HandDetector.kt)
* Configurar `HandLandmarker` para detectar hasta 2 manos (`setNumHands(2)`).
* Instanciar `PoseLandmarker` para extraer landmarks del tronco superior (hombros, codos).

#### [MODIFY] [LandmarkNormalizer.kt](file:///c:/Users/jeffe/Documents/Proyectos/ModeloPrototipo/frontend/app/src/main/java/com/lsec/translator/ml/LandmarkNormalizer.kt)
* Actualizar el algoritmo para normalizar la posición de los landmarks de ambas manos y codos utilizando los hombros como punto de referencia global `(0,0,0)`.

---

### Componente de ML (Python Backend)

#### [MODIFY] [spoter_model.py](file:///c:/Users/jeffe/Documents/Proyectos/ModeloPrototipo/backend/hand_model/spoter_model.py)
* Modificar la capa lineal de entrada `self.proj` del `SignTransformer` para aceptar el nuevo tamaño de vector tridimensional expandido (ej. 150 dimensiones).

#### [MODIFY] [landmarks.py](file:///c:/Users/jeffe/Documents/Proyectos/ModeloPrototipo/backend/hand_model/landmarks.py)
* Actualizar `LandmarkExtractor` para usar la solución de Pose + Hands de MediaPipe en Python y exportar el nuevo formato de landmarks.

#### [MODIFY] [reextract_sequences_invariant.py](file:///c:/Users/jeffe/Documents/Proyectos/ModeloPrototipo/backend/scripts/reextract_sequences_invariant.py)
* Adaptar el pipeline de re-extracción para procesar de nuevo todos los videos crudos de LSC50.

---

## Plan de Verificación

1. **Extracción y Validación de Datos**:
   * Ejecutar la extracción holística en un video de prueba y validar que deje archivos `.npy` con las dimensiones correctas.
2. **Entrenamiento de Prueba**:
   * Correr un entrenamiento de juguete de 2 épocas para verificar que el `SignTransformer` asimile el nuevo tamaño de entrada sin errores de dimensiones en PyTorch.
3. **Exportación e Inferencia**:
   * Exportar el modelo entrenado a ONNX e importarlo en Android Studio para verificar la compatibilidad de carga en la aplicación móvil.
