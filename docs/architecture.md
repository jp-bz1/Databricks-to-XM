# Arquitectura de la solución

Pregunta de negocio: para cada combinación activa de operador de red, mercado de comercialización, tipo de mercado y sector CIIU, ¿cuál es la demanda de energía esperada para los próximos 7 días?

Patrón de solución: **batch diario + machine learning**. Los datos llegan por archivo con retraso variable (5 a 205 días) y el pronóstico se necesita una vez al día. No hay caso para streaming.

## Capas y contratos

Llena una fila por tabla. Reemplaza cada `<…>`. El dueño es un rol de XM, no una persona.

| Capa | Tabla | Grano (qué es una fila) | Dueño | Frescura | Garantías |
|---|---|---|---|---|---|
| Bronce | `bronze_energia.demanda_raw` | Un registro publicado, tal como llegó del archivo fuente. | Ingeniería de Datos XM. | Se actualiza cuando llega un nuevo archivo. | Conserva los datos originales, sus republicaciones y los metadatos de ingesta sin modificar ni eliminar registros. |
| Plata | `silver_energia.demanda_diaria` | Una fila por serie-día con demanda real y pérdidas. | Ingeniería de Datos XM. | Actualización diaria después de procesar Bronce. | Contiene la versión vigente, tipos y nombres normalizados, y registros que cumplen las reglas de calidad. |
| Plata | `silver_energia.dim_ciiu` | Una fila por clasificación industrial CIIU. | Gobierno de Datos XM. | Se actualiza cuando cambia o aparece una clasificación CIIU. | Cada código CIIU es único, no nulo y tiene una descripción estandarizada. |
| Oro | `gold_energia.features_demanda_diaria` | Una fila por serie-día con las variables utilizadas por el modelo. | Analítica y Ciencia de Datos XM. | Actualización diaria después de Plata. | Las variables usan solamente información disponible hasta la fecha analizada, sin fuga de información futura. |
| Oro | `gold_energia.pronostico_demanda` | Una fila por serie, fecha de pronóstico y horizonte de 1 a 7 días. | Analítica y Ciencia de Datos XM. | Se genera diariamente para los siguientes 7 días. | Cada pronóstico identifica la versión del modelo, la ejecución y la fecha de generación. |

## Llave de serie

`codigo_sic_agente`, `mercado_comercializacion`, `tipo_mercado`, `clasificacion_industrial`. 355 combinaciones activas.

## Decisiones de diseño derivadas de la exploración (lab 2)

| Hecho | Decisión | Dónde se implementa |
|---|---|---|
| Formato largo (2 filas por serie-día) | Consolidar DdaReal y PerdidasEnergia en una fila por serie-día. Si falta alguna variable, enviar el registro a una zona de revisión sin imputar cero. | Plata (clase 6) |
| Publicación con retraso variable y republicaciones | Bronce acumula todas las publicaciones sin sobrescribir. Plata selecciona como vigente el registro con la FechaPublicacion más reciente. | Bronce (clase 4) / Plata (clase 6) |
| Series incompletas (4 de 355) | No rellenar con cero los días faltantes. Conservar los datos disponibles, marcar las series inactivas y excluir del modelo las que no tengan historial suficiente. | Plata / features (clase 7) |
| Regulado vs. no regulado | Entrenar un modelo por tipo de mercado, debido a las diferencias de escala, participación energética, diversidad sectorial y cantidad de series entre Regulado y No Regulado. Evaluar ambos modelos con una métrica normalizada como MASE. | Modelo (clase 9) — ver ADR-001 |
| Ceros (299 serie-días) | Conservar las filas con demanda cero y generar una advertencia de calidad para revisión. No eliminarlas ni modificarlas automáticamente. | Reglas de calidad (clase 5) |
| Pérdidas ≤ demanda | Aplicar perdidas_kwh <= demanda_real_kwh como regla de calidad y pronosticar únicamente la demanda, conservando las pérdidas como variable de análisis y control. | Reglas de calidad (clase 5) |

## Diagrama

```
DemandaPerdidas.xlsx ──▶ [volumen raw] ──▶ bronze_energia.demanda_raw
                                                │
                                                ▼  (pipeline declarativo, reglas de calidad)
                                     silver_energia.demanda_diaria ◀── silver_energia.dim_ciiu
                                                │
                                                ▼  (job de features)
                                  gold_energia.features_demanda_diaria
                                                │
                                                ▼  (job de inferencia, modelo en UC)
                                     gold_energia.pronostico_demanda ──▶ tablero / app
```

En Free Edition todo vive en el catálogo `workspace`; desde la clase 3, en `dev`, `qa` y `prod`. En Azure Databricks cada capa se registra como external location sobre ADLS Gen2.

## ADRs relacionadas

- [ADR-001 — Granularidad y horizonte del pronóstico](adr/ADR-001-granularidad-horizonte.md)
