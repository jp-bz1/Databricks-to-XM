# Lab 9 — Experimentos, evaluación honesta y modelo registrado

**Duración:** 30 min (más el notebook guiado) · **Modalidad:** individual, en tu propio fork · **Rama:** `feature/c09-mlflow`

## Objetivo

Construir el experimento `demanda_7d` con varios modelos de varios tipos, todos rastreados de la misma forma; compararlos contra un baseline con métricas por segmento y validación temporal; registrar dos candidatos en Unity Catalog con `@champion` y `@challenger`. Abre el hito 5.

## Antes de empezar

| Requisito | Si no lo tienes |
|---|---|
| Lab 8 terminado: `workspace.c01_<usuario>.features_demanda_diaria` con 24 columnas | Si la tabla tiene `perdidas_kwh`, `DROP TABLE` y vuelve a correr el lab 7 |
| Git folder en `main` actualizado | GitHub: **Sync fork** → Databricks: Git → **Pull** |
| Rama nueva | Git → Create branch → `feature/c09-mlflow` |

## Pasos

Abre `clases/c09-mlflow/notebooks/lab09_mlflow_registro`. Widget `usuario`. La primera celda instala `lightgbm`.

| Sección | Qué haces | Qué debes ver |
|---|---|---|
| 1–2 | Leer el ciclo; corte temporal | ~54 mil filas de entrenamiento hasta el 30 jun; ~6.300 de validación 1–18 jul |
| 2 TODO 1 | `wape` en `src/xm_demanda/model/metricas.py` | Los tres `assert` pasan |
| 3 | Primer run a mano: `naive_estacional` | WAPE ≈ 0,048. Abre el run: parámetros, métricas, dataset con versión, artefacto `modelo/` con firma |
| 4 | `lightgbm_global` | WAPE ≈ 0,050: **no vence al baseline** |
| 5 | Tabla de runs y error por día | Lunes ≈ 0,09–0,14: los dos lunes festivos |
| 6 TODO 2 | `NaiveFestivo.fit` en `src/xm_demanda/model/modelos.py` | WAPE ≈ 0,046; lunes baja a ≈ 0,08 |
| 7 | Walk-forward con runs anidados | Promedio: lightgbm 0,048 · naive_festivo 0,054 · naive 0,054 |
| 8 | Run con fuga | WAPE ≈ 0,024, tag `fuga = true`. **Pregunta 1** |
| 9 | Clasificador y KMeans | AUC ≈ 0,81 · silhouette ≈ 0,50 |
| 10 | Registro y alias; inferencia con `@champion` | Dos versiones en Catalog Explorer. **Pregunta 2** |
| `verificar()` | 10 comprobaciones en OK | Commit `feat(c09): experimento demanda_7d, baselines, lightgbm y modelo registrado` → push → PR → CI → merge |

Los valores son de referencia (con la tabla de oro de la docente); los tuyos pueden diferir en el tercer decimal.

## Criterios de aceptación

- [ ] `wape` y `NaiveFestivo.fit` implementadas; `tests/unit/test_model.py` sin skip
- [ ] Experimento con los 7 runs: `naive_estacional`, `lightgbm_global`, `naive_festivo`, `walk_forward` (+3 hijos), `lightgbm_fuga`, `clasificador_caida`, `kmeans_perfiles`
- [ ] Cada run de pronóstico con `wape`, `mae_gwh`, `wape_regulado`, `wape_no_regulado`, dataset con versión Delta y modelo con firma
- [ ] `workspace.c01_<usuario>.demanda_7d` con `@champion` y `@challenger` en versiones distintas
- [ ] Las dos preguntas respondidas
- [ ] `verificar()` en OK; PR en verde

**Hito 5 abierto** (se cierra en la clase 11 con el pronóstico publicado).

## Comparar con la solución de referencia

`https://github.com/<docente>/Databricks-to-XM/compare/s09...<tu_usuario>:Databricks-to-XM:main`

```bash
git fetch upstream --tags
git diff s09 -- src/xm_demanda/model clases/c09-mlflow/notebooks
```

## Extensión opcional

1. **Un festivo como feature, no como regla.** Agrega a oro `es_festivo_objetivo` (si `fecha + 7` es festivo) y reentrena `lightgbm_global`. ¿Vence a `naive_festivo` el lunes?
2. **Ajuste de hiperparámetros con runs anidados.** Un run padre `busqueda_lgbm` con 6 hijos variando `num_leaves` y `learning_rate`; elige por WAPE de walk-forward, no del corte único.
3. **`mlflow.evaluate`.** Evalúa `@champion` sobre la validación con `model_type="regressor"` y compara las métricas que calcula con las tuyas.
