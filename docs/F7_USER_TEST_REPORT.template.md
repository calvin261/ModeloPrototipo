# F7 — Reporte de Validación con Usuarios Reales

> Reporte generado por `backend/scripts/analyze_f7_results.py` a
> partir de los CSV de `SignTestLogger.kt`. La plantilla define
> los placeholders `{{...}}` que el script rellena.

## Resumen ejecutivo

- **Testers reclutados:** {{N_USERS}}
- **Trials Ronda 1 (sin calibración):** {{N_TRIALS_R1}}
- **Trials Ronda 2 (con calibración):** {{N_TRIALS_R2}}
- **Accuracy global Ronda 1:** **{{ACC_R1}}** — objetivo ≥ {{TARGET_NO_CALIB}} → {{VEREDICT_R1}}
- **Accuracy global Ronda 2:** **{{ACC_R2}}** — objetivo ≥ {{TARGET_WITH_CALIB}} → {{VEREDICT_R2}}
- **Mejora relativa por calibración:** {{REL_IMPROVEMENT}}
- **Referencia interna** (SignTransformer LOSO 18 signers, dataset LSC50): {{REFERENCE_LOSO}}

### Tabla resumen

{{SUMMARY_TABLE}}

### Veredicto de la fase

- Ronda 1 vs objetivo ≥ {{TARGET_NO_CALIB}}: **{{VEREDICT_R1}}**
- Ronda 2 vs objetivo ≥ {{TARGET_WITH_CALIB}}: **{{VEREDICT_R2}}**
- Si la Ronda 2 cumple y la Ronda 1 queda por debajo del objetivo, documentar
  explícitamente: la calibración de 30-60 s es **obligatoria** para alcanzar
  el umbral de despliegue, lo cual se incorpora al manual de usuario.

---

## 1. Accuracy por tester

{{USER_TABLE}}

**Interpretación.** Un tester con Δ < -2% en la ronda calibrada
es señal de que la calibración **empeoró** su embedding — caso
raro pero posible si el tester se movió mucho entre rondas. Un
tester con Δ > +15% se beneficia mucho de la personalización.

## 2. Accuracy por seña (agregada sobre los 5 testers)

{{SIGN_TABLE}}

**Interpretación.** Las señas con <70% en Ronda 2 son candidatas
a reentrenamiento o a una versión 2 del vocabulario. Las señas
con >95% en Ronda 1 y bajan en Ronda 2 son candidatas a
**revisar la calibración** (quizá el tester no entendió
"ejemplo claro" y dio variantes inconsistentes).

## 3. Matriz de confusión — Ronda 1 (sin calibración)

{{CM_R1}}

## 4. Matriz de confusión — Ronda 2 (con calibración)

{{CM_R2}}

**Interpretación.** Comparar ambas matrices diagonal-fila:
- Confusiones que **desaparecen** en Ronda 2 son las que la
  calibración resolvió (típicamente errores de "seña parecida
  en el dataset LSC50, distinta en el tester").
- Confusiones que **persisten** en Ronda 2 son errores de
  vocabulario o de UI: el tester probablemente firmó algo
  distinto a lo que se le pidió.
- Confusiones **nuevas** en Ronda 2 son las que la calibración
  **introdujo** (embedding se alejó del cluster correcto);
  investigar caso por caso.

## 5. Latencia

- **Ronda 1:** mediana {{MED_LAT_R1}} ms, p95 {{P95_LAT_R1}} ms
- **Ronda 2:** mediana {{MED_LAT_R2}} ms, p95 {{P95_LAT_R2}} ms

**Objetivo del README:** <500 ms end-to-end. La latencia debería
ser **sustancialmente menor** (~80-200 ms en Infinix GT30 Pro con
SignTransformer exportado a TFLite). Si se observa >500 ms, es
bug — reportar.

## 6. Conclusiones y siguientes pasos

<!-- Llenar manualmente después de revisar las tablas. -->

- ¿Se cumple el objetivo global? [Sí/No/Parcial]
- ¿Algún tester es outlier? [Sí/No — cuál]
- ¿Alguna seña requiere retirar del vocabulario? [Sí/No — cuáles]
- ¿La calibración aporta suficiente valor? [Sí/No]
- ¿Listo para escalar a 15-20 testers? [Sí/No]
- ¿Cambios al modelo antes de Fase 8? [Sí/No — cuáles]

## 7. Datos crudos

- CSVs fuente: `reports/f7/P0X_round{1,2}.csv`
- Comando para regenerar este reporte:
  `cd backend && python scripts/analyze_f7_results.py \\
      --csv-dir reports/f7 \\
      --output ../docs/F7_USER_TEST_REPORT.md`
