# ADR-001: Granularidad y horizonte del pronóstico

- **Estado:** aceptada
- **Fecha:** 2026-09-27
- **Autor:** Juan Pablo Bedoya Zapata

## Contexto

La solución cuenta con 206 días de datos diarios y 355 series activas. Los datos llegan mediante archivos con retrasos de publicación entre 5 y 205 días. XM requiere estimar la demanda de energía de cada serie para los próximos 7 días. Las series reguladas y no reguladas presentan escalas diferentes y algunas tienen información incompleta.

## Decisión

Se utilizará granularidad diaria, un horizonte de pronóstico de 7 días y un modelo global para las 355 series. El modelo incluirá TipoMercado, ClasificacionIndustrial y las demás características de cada serie como variables explicativas, con el propósito de aprender las diferencias entre los mercados Regulado y No Regulado. Su desempeño se evaluará separadamente por tipo de mercado mediante una métrica normalizada como MASE.

## Alternativas consideradas

- **Un modelo diferente por tipo de mercado:** se descarta inicialmente porque exige entrenar, versionar y mantener varios modelos, además de reducir la información compartida durante el entrenamiento.
- **Modelo independiente por serie:** se descarta porque requeriría mantener 355 modelos y algunas series no tienen suficiente historial.

## Consecuencias

El modelo global simplifica el entrenamiento, versionamiento y mantenimiento, y permite aprovechar la información conjunta de las 355 series. Como riesgo, las series reguladas de mayor escala podrían dominar el aprendizaje. Por esta razón, el desempeño se evaluará separadamente para Regulado y No Regulado mediante una métrica normalizada. Si algún segmento presenta resultados deficientes, se reconsiderará utilizar modelos separados por tipo de mercado.
