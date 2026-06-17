# F7 — Protocolo de Validación con Usuarios Reales

> Test de campo con **5 testers humanos** usando el Infinix GT30 Pro
> con la app LSEC Translator. Vocabulario: las 12 señas dinámicas de
> `backend/data/sequences/vocabulary.json` (adios, ayuda, bano, bien,
> como_estas, gracias, hola, mal, no, perdon, por_favor, si).
>
> **Objetivos cuantitativos:**
> - Top-1 sin calibración ≥ **90%**
> - Top-1 con calibración ≥ **95%**
>
> **Referencia interna (LOSO 18 signers, dataset LSC50):** el
> SignTransformer alcanza **87.39%** de media sin personalización,
> lo que sitúa el 90% en un rango alcanzable solo si la sesión
> del usuario nuevo es más limpia que el dataset (lo cual es
> esperable: el tester está frente a cámara, iluminado, a distancia
> fija). La calibración debe empujarlo por encima del 95%.

---

## 1. Pre-test setup (responsable: investigador)

Antes de citar al tester, dejar el dispositivo en estado óptimo:

- **Batería:** cargar a **100%**. La app con la cámara + inferencia
  TFLite + logging CSV consume ~5%/10 min. Con batería baja el
  teléfono puede throttlear la CPU y la latencia medida deja de
  reflejar el uso real.
- **Iluminación:** habitación con luz difusa frontal (ventana grande
  o lámpara de escritorio a 1.5 m del sujeto, **no** luz desde
  atrás porque silueta la mano). Verificar que no haya sombras
  fuertes sobre la mano.
- **Distancia cámara-sujeto:** ~**1 m** medido con cinta (no a ojo).
  A 0.5 m la mano sale recortada a 30 fps; a 1.5 m los landmarks
  pierden precisión sub-pixel.
- **Entorno:** silencioso (sin TV, sin música). Evita que el
  tester se desconcentre y reduce varianza en el ritmo de las
  señas.
- **Fondo:** pared lisa, color uniforme. Evitar fondos con
  patrones que confunden al detector de manos.
- **App:** versión compilada con el modelo **actual** del
  registry (`backend/models/exports/registry.json`), no un build
  de desarrollo. Verificar que la app abre sin errores y que
  el modelo se carga (el log `Loaded X centroids` no debe
  aparecer antes de calibrar).
- **Almacenamiento:** al menos **200 MB libres** en
  `getExternalCacheDir()` para los CSV. Borrar logs de
  testers anteriores.
- **Tester ID:** asignar `P01`..`P05` antes de la sesión. Anotar
  edad, sexo y mano dominante (ver §6).

---

## 2. Consentimiento (informal, sin IRB)

Leer en voz alta antes de empezar, y **solo si el tester acepta**
continuar:

> "Vamos a probar un prototipo de reconocimiento de lengua de
> señas. La app va a guardar en este celular un archivo CSV con
> tus señas, qué palabra se te pidió, qué predijo el modelo, y
> cuánto tardó. **No se guarda tu nombre, tu cara, ni tu voz.**
> Cuando terminemos, puedo borrar el archivo del celular si me
> lo pides. ¿Te parece bien?"

Es un consentimiento **informal, no legal** — no se firma nada,
no hay comité de ética. Es una sesión de prueba técnica entre
el equipo del proyecto y un voluntario. El investigador anota
"consent=si/no" en su bitácora personal.

Si el tester dice "no" o se muestra incómodo, se agradece y se
cancela la sesión. **No hay compensación económica prometida**;
si se da un refrigerio, se documenta aparte como "anexo" y no
como pago.

---

## 3. Flujo por usuario (~25 min totales)

| Paso | Duración | Descripción |
|------|----------|-------------|
| a. Intro a la app | 1 min | Mostrar la app, explicar qué hace (1 frase). |
| b. Práctica (no scored) | 2 min | El tester hace 3-4 señas libre para familiarizarse. |
| c. **Ronda 1 (sin calibración)** | ~5 min | 36 señas (12 palabras × 3 reps), orden aleatorio. |
| d. Calibración | 30-60 s | 3 ejemplos por seña (36 señas de calibración). |
| e. **Ronda 2 (con calibración)** | ~5 min | Mismas 36 señas, nuevo orden aleatorio. |
| f. Encuesta subjetiva | 2 min | 5 preguntas, escala 1-5. |
| g. Cierre y agradecimientos | 1 min | Ofrecer borrar el CSV. |

### a. Intro a la app (1 min)

Mostrar la pantalla principal. Decir literalmente:

> "Esta app reconoce 12 señas colombianas. Te voy a pedir que
> hagas señas frente a la cámara. La app muestra en pantalla qué
> cree que dijiste. Si te equivocas, no importa, solo vuelve a
> hacerla cuando te lo pida."

No explicar el modelo, no decir "machine learning", no decir
"transformer". Evitar sesgo de expectativa.

### b. Práctica (2 min, NO scored)

Decir:

> "Primero vamos a hacer 3 señas de práctica, solo para que
> pruebes la cámara. Estas **no se guardan**."

Pedir: hola, gracias, adios (en ese orden, hablado, no signado).
Si el tester se ve nervioso, repetir.

### c. Ronda 1 — sin calibración (≈5 min)

- Pulsar **Iniciar Test** en `TestSessionScreen` (modo `round=1`).
- Por cada seña:
  1. La app muestra la palabra objetivo (texto grande).
  2. El tester espera ~1.5 s (la app muestra countdown).
  3. El tester hace la seña cuando aparece "AHORA".
  4. **Latencia empieza aquí** — el momento "AHORA" es el
     `t_start`.
  5. La app muestra la predicción (texto + confianza).
  6. **Latencia termina aquí** — el momento en que aparece
     la predicción en pantalla es el `t_end`.
  7. Pulsar **Siguiente** (también se acepta timeout 4 s).
- Total: 36 trials. Orden aleatorio (mismo seed por tester para
  reproducibilidad entre rondas: ver §5).
- **NO hay feedback correctivo** durante la prueba. La app solo
  dice "OK" brevemente. Decir "no te corrijo para no influirte".
- Anotar manualmente cualquier evento raro en la bitácora
  (tester se distrae, se le cae el teléfono, etc.).

### d. Calibración (30-60 s)

- Pulsar **Iniciar Calibración**.
- Para cada una de las 12 señas, el tester hace **3 ejemplos**.
  Total: 36 grabaciones de calibración.
- La app usa el `CentroidClassifier` existente para extraer el
  embedding penúltima-capa de cada ejemplo y promediar.
- Si una seña falla 3 veces (mano no detectada), saltarla y
  anotarlo. **El tester no debe repetir la calibración** por
  un fallo aislado.
- Duración típica: 45 s. Si pasa de 60 s, detener y registrar.

### e. Ronda 2 — con calibración (≈5 min)

Idéntica a la Ronda 1, pero con `round=2`. Orden aleatorio
**distinto** de la Ronda 1 (mismo seed + offset). Aplican las
mismas reglas (sin feedback correctivo).

### f. Encuesta subjetiva (5 preguntas, escala 1-5)

Anotar en la bitácora las respuestas del tester:

1. **Naturalidad**: ¿qué tan natural se sintió hacer las señas
   frente a la app? (1 = forzado, 5 = como hablar con alguien)
2. **Latencia percibida**: ¿qué tan rápido sentiste que la app
   respondía? (1 = muy lento, 5 = inmediato)
3. **Ayuda de la calibración**: ¿sentiste que después de
   calibrar la app te entendía mejor? (1 = igual, 5 = muchísimo
   mejor)
4. **Señas problemáticas**: ¿hubo alguna seña que sentiste que
   la app nunca entendía? (anotar texto libre: "gracias",
   "como_estas", "ninguna", etc.)
5. **Uso diario**: ¿usarías esta app en tu vida diaria?
   (1 = nunca, 5 = todos los días)

### g. Cierre

- Ofrecer borrar el CSV (es su derecho). Por defecto, **se
  conserva** para el análisis, anonimizado como `P0X.csv`.
- Agradecer.

---

## 4. Recolección de datos (consentimiento y privacidad)

**Lo que SÍ se guarda** (formato CSV, ver §5):

- `tester_id` (P01-P05)
- `round` (1 o 2)
- `rep` (1-3)
- `prompted_sign` (string)
- `predicted_sign` (string o `<none>` si el modelo no detectó)
- `confidence` (float 0-1)
- `latency_ms` (int, medido de "AHORA" a predicción en pantalla)
- `timestamp_ms` (epoch ms del momento "AHORA")
- `seed` (int, para reproducibilidad del orden)

**Lo que NO se guarda:**

- Nombre, cédula, edad exacta, foto, video, audio, GPS, contactos.
- Edad se anota **por rangos** (20-30, 31-45, 46-60) en la
  bitácora del investigador, no en el CSV.
- Mano dominante **sí** va en el CSV (atributo no-PII).

---

## 5. Orden aleatorio reproducible

Para que las dos rondas de un tester no compartan orden (y
para que entre testers la distribución sea comparable), se usa
un seed determinístico:

```kotlin
val orderR1 = shuffle(vocabulary.repeated(3), seed = testerId.hashCode())
val orderR2 = shuffle(vocabulary.repeated(3), seed = testerId.hashCode() xor 0xCAFE)
```

Ambos órdenes se guardan en el CSV al inicio de cada ronda
(columna `seed`), para auditoría posterior.

---

## 6. Plantilla de reclutamiento

**Perfil buscado:** 5 voluntarios, sin importar si conocen LSC
o no, pero con habilidad motora normal.

| Slot | Sexo | Edad (rango) | Mano dominante | Experiencia con LSC |
|------|------|--------------|----------------|---------------------|
| P01  | M    | 20-30        | Derecha        | Ninguna             |
| P02  | F    | 20-30        | Izquierda      | Ninguna             |
| P03  | M    | 31-45        | Derecha        | Ha visto señas      |
| P04  | M    | 46-60        | Derecha        | Ninguna             |
| P05  | F    | 31-45        | Derecha        | Interprete amateur  |

**Criterios:**

- 3 hombres + 2 mujeres (requisito del advisor).
- 2 zurdos + 3 diestros (P02 y un cuarto zurdo adicional —
  ajustar tabla si se recluta diferente).
- Mix de **experimentados y novatos**: 1-2 con conocimiento
  previo de LSC, 3-4 sin conocimiento (el "uso diario" mide
  el caso realista).
- Edad 20-60, dentro de los 3 rangos.
- Sin lesiones en la mano dominante (el tester lo declara
  verbalmente al inicio).
- Sin trastornos neurológicos que afecten el control motor
  fino (declaración verbal).

**Reclutamiento informal:** pasar la voz en el programa de
ingeniería, en grupos de estudio, y en una asociación local
de sordos. **No se firma contrato, no se paga, no se pide
documento de identidad.** Si el voluntario quiere conocer
los resultados, se le envía el reporte final por WhatsApp.

**Anuncio sugerido** (copiar y pegar):

> "Busco 5 personas (20-60 años) para probar un prototipo
> de app que reconoce señas colombianas. Dura 25 minutos,
> sin papeles, sin paga (hay tinto). Necesito gente diestra
> y zurda, con y sin experiencia en lengua de señas.
> Escríbeme al [WhatsApp]."

---

## 7. Materiales que debe llevar el investigador

- Infinix GT30 Pro cargado al 100%
- Cinta métrica (para los 1 m)
- Lámpara de escritorio (respaldo de iluminación)
- Bitácora impresa con la tabla del §6 y las 5 preguntas
  de la encuesta
- Cargador portátil (por si la sesión se alarga)
- Este documento impreso

---

## 8. Métricas que se calculan después (en `analyze_f7_results.py`)

- Accuracy por tester (Ronda 1 y Ronda 2 separadas)
- Accuracy por seña (agregada sobre los 5 testers)
- Matriz de confusión 12×12 (Ronda 1 y Ronda 2)
- Latencia mediana y p95 (por ronda)
- Mejora relativa calibración: `(acc_R2 - acc_R1) / acc_R1`
- Veredicto GO/NO-GO vs objetivos §0 (90% / 95%)

Ver plantilla: `docs/F7_USER_TEST_REPORT.template.md`.
