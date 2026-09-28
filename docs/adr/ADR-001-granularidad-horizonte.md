# ADR-001: Granularidad y horizonte del pronóstico

- **Estado:** aceptada
- **Fecha:** 2026-09-27
- **Autor:** Juan Pablo Bedoya Zapata

## Contexto

La solución cuenta con 206 días de datos diarios y 355 series activas. Los datos llegan mediante archivos con retrasos de publicación entre 5 y 205 días. XM requiere estimar la demanda de energía de cada serie para los próximos 7 días. Las series reguladas y no reguladas presentan escalas diferentes y algunas tienen información incompleta.

## Decisión

Se utilizará granularidad diaria, un horizonte de pronóstico de 7 días y un modelo diferente para cada tipo de mercado. Esta separación se adopta porque el mercado Regulado concentra el 69,1 % de la energía en solo 29 series, mientras que el No Regulado contiene 326 series de menor escala y mayor diversidad industrial.

## Alternativas consideradas

- **Un modelo global para todas las series:** se descarta porque las diferencias de escala y comportamiento entre Regulado y No Regulado podrían provocar que el modelo quede dominado por las series reguladas de mayor demanda.
- **Modelo independiente por serie:** se descarta porque requeriría mantener 355 modelos y algunas series no tienen suficiente historial.

## Consecuencias

La separación permite que cada modelo aprenda las características particulares de su tipo de mercado y evita que las grandes series reguladas dominen el entrenamiento. Como contrapartida, se deberán entrenar, evaluar, versionar y mantener dos modelos. Queda pendiente comprobar mediante métricas si esta separación mejora el desempeño frente a un modelo global.
