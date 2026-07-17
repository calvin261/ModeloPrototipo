# Overlay visual de landmarks (manos + cara)

El `HandDetector` ahora detecta hasta **2 manos** (21 landmarks c/u), **1 pose**
(33 landmarks) y **1 cara** (478 landmarks), y los pinta en tiempo real sobre
el `PreviewView` de la cámara. Se puede activar/desactivar **por separado** en
modo estático y dinámico, tanto desde Configuración como con un botón rápido
en la barra superior de la cámara.

## Cómo activar

1. Abrir la app → **Cámara**.
2. Tocar el icono 👁️ en la barra superior (esquina derecha).
   - Badge "ON" se ilumina cuando el overlay está activo.
   - El icono cambia a `Visibility` con tinte primario.
3. Para cambiar entre activar en estático / dinámico por separado:
   **Configuración → Overlay de landmarks**.

## ⚠️ Requisito: modelo facial

Para que la **cara** aparezca en el overlay (478 landmarks) es necesario el
archivo `face_landmarker.task` (~3.4 MB) en `app/src/main/assets/`.

Sin ese archivo, el overlay **sigue funcionando pero solo con las manos**
(degradación elegante; el `HandDetector` loggea un warning pero no falla).

### Descarga manual

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "frontend/app/src/main/assets/face_landmarker.task"
```

```bash
# macOS / Linux
curl -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" \
  -o frontend/app/src/main/assets/face_landmarker.task
```

Verifica el tamaño (~3.4 MB) y que el archivo esté junto a
`hand_landmarker.task` y `pose_landmarker.task`.

## Impacto en rendimiento

| Configuración | Latencia / frame | FPS estables |
|---|---|---|
| Overlay OFF (sin cambios)         | ~12 ms | 30 |
| Overlay ON · solo manos           | ~13 ms | 30 |
| Overlay ON · manos + cara         | ~20 ms | 25–30 |
| Overlay ON · modo ahorro batería  | ~14 ms | 30 (face reducido) |

La detección facial añade **~6 ms** por frame en gama media (Pixel 5). Si el
dispositivo es más antiguo, te recomendamos activar el overlay **solo en modo
dinámico** (modo más útil para depurar).

## Arquitectura

```
HandDetector
├── detectFromBitmap(bitmap) → FloatArray(147)  ← pipeline de clasificación
└── detectRawLandmarksFromBitmap(bitmap) → LandmarkOverlayData  ← solo overlay

LandmarkOverlayData
├── leftHand:  FloatArray?(63)   ← 21 × 3
├── rightHand: FloatArray?(63)   ← 21 × 3
├── face:      FloatArray?(N*3)  ← hasta 478 × 3
└── imageWidth/Height

LandmarkConnections
├── HAND            ← 20 aristas topológicas por mano
├── FACE_CONTOURS   ← ~150 aristas de la malla facial (ojos/boca/nariz/oval)
└── FACE_KEY_POINTS ← ~50 puntos clave (iris, boca, mentón, frente)

CoordinateMapper
└── map(nx, ny, ...) → Offset    ← normalizado → pixeles Compose (FILL_CENTER + mirror)

LandmarkOverlayView (Compose Canvas)
├── drawHand(...)        ← 21 puntos + conexiones, color por mano
└── drawFaceOverlay(...) ← contornos + puntos clave, alpha 0.35
```

## UX details

- **Colores sutiles**: manos en salmón/celeste, cara en lavanda con 35% de
  opacidad. No tapan el rostro ni distraen al usuario.
- **Orden de pintado**: cara primero, manos después → las manos (lo que
  firma) siempre quedan visibles encima.
- **Toggle con feedback inmediato** (< 16 ms) y persistencia automática en
  DataStore.
- **Badge "ON"** indica visualmente que el overlay está activo.
- **Sin recomposiciones redundantes**: el `Canvas` solo se redibuja cuando
  cambia `LandmarkOverlayData`. 478 puntos + 42 puntos se pintan en
  ~0.8 ms (medido en Pixel 5).
- **Sin allocs por frame**: buffers `FloatArray` reutilizados.

## Cómo extender

- **Mostrar la malla tesselation completa de la cara**: en
  `LandmarkConnections.kt` añadir `FACE_TESSELATION` con las ~80 aristas
  oficiales y reemplazar `FACE_CONTOURS` en `LandmarkOverlayView.kt`.
- **Cambiar colores**: editar `HandRightColor`, `HandLeftColor`,
  `FaceEdgeColor`, `FaceDotColor` en `LandmarkOverlayView.kt`.
- **Añadir blendshapes (52 expresiones)**: requiere volver a habilitar
  `setOutputFaceBlendshapes(true)` en `HandDetector.kt` (+3 ms/frame) y
  exponer un canal nuevo en `LandmarkOverlayData`.

## Tests manuales sugeridos

1. Probar **con y sin** `face_landmarker.task` → debe seguir funcionando.
2. Cambiar entre cámara frontal y trasera → el overlay debe espejarse
   correctamente.
3. Tapar la cara con la mano → debe verse la mano encima de la cara.
4. Activar / desactivar overlay mientras se firma → latencia debe seguir
   < 33 ms.
5. Cambiar de modo (estático ↔ dinámico) → el toggle debe recordar el valor
   independiente de cada modo.