from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

from replica_inpp.dominio.calculo.base import (
    CalculadorBase,
    _construir_diagnostico,
    _laspeyres_por_grupo,
    _recortar_series_fecha,
    _rellenar_dato_serie_faltante,
    _validar_serie_cubre_grupo,
)
from replica_inpp.dominio.errores import CanastaSinGenericos, InvarianteViolado, VersionNoCoincide
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.tipos import (
    AGREGACION_A_COLUMNA,
    AGREGACION_A_NOMBRE,
    AGREGACIONES_VALIDAS,
    CODIGO_PETROLEO_CRUDO,
    RUBRO_A_COLUMNA_PESO,
    RUBROS_POR_RECORTE,
    SECTORES_MERCANCIAS,
    TIPO_INPP,
    ManifestCalculo,
)

_COLUMNA_SECTOR = "codigo sector"

# Texto bare ("1111", igual al código, sin nombre real) vs combinado ("1111
# Cultivo de cereales..."). Ver `LectorCanastaCsv._separar_codigo_jerarquia`:
# la columna sin prefijo "codigo " trae el texto original tal cual vino del
# CSV fuente, uniforme en toda la columna (o 100% combinada o 100% bare).
_PATRON_BARE = re.compile(r"^\d+$")


class LaspeyresDirecto(CalculadorBase):
    """Etapa 2 (Laspeyres) sin encadenar — la canasta 2019 corre las dos etapas así.

    A diferencia de INPC (agrupa por 1 solo eje, `tipo`), acá se agrupa por 2 ejes
    independientes: `agregacion` (nivel SCIAN, `"INPP"` general, o
    `"MERCANCIAS_SERVICIOS"`) y `rubro` (columna de peso/destino de producción —
    ver `RUBRO_A_COLUMNA_PESO`). Cada corrida cubre una sola combinación
    `(version, agregacion, rubro)`; combinar varias corridas en un único
    `ResultadoIndice` (todos los rubros a la vez) es una función aparte,
    todavía sin implementar (ver docstring de `ResultadoIndice.manifiesto`).
    """

    def calcular(
        self,
        canasta: CanastaINPP,
        serie: SerieNormalizada,
        agregacion: str,
        rubro: str | None = None,
        sin_petroleo: bool = False,
    ) -> ResultadoIndice:
        fecha = datetime.now()
        ruta_canasta = canasta.df.attrs.get("origen")
        ruta_serie = serie.df.attrs.get("origen")

        # `SerieNormalizada` no modela `version` como campo propio -- solo
        # `cargar_serie` la deja en `df.attrs["version"]`. Si no está (serie
        # construida a mano, sin pasar por `cargar_serie` -- el caso de todos
        # los tests sintéticos), no hay nada contra qué comparar y se omite la
        # validación, mismo comportamiento de siempre. Si SÍ está y no coincide
        # con `canasta.version`, es una mezcla real de vintages (encontrado con
        # archivos reales: canasta 2019 + serie 2012 calculó sin fallar y dio
        # un valor con apariencia válida).
        version_serie = serie.df.attrs.get("version")
        if version_serie is not None and version_serie != canasta.version:
            raise VersionNoCoincide(
                f"canasta versión {canasta.version} no coincide con la versión de la serie "
                f"({version_serie}) -- canasta y serie deben ser de la misma versión."
            )

        agregacion = agregacion.upper()
        if agregacion not in AGREGACIONES_VALIDAS:
            raise InvarianteViolado(
                f"agregacion='{agregacion}' no es válida; ver tipos.py::AGREGACIONES_VALIDAS"
            )
        # Normaliza abreviatura ("R") a nombre canónico ("RAMA") — TIPO_INPP y
        # MERCANCIAS_SERVICIOS ya son su propio nombre canónico, no están en el mapeo.
        agregacion = AGREGACION_A_NOMBRE.get(agregacion, agregacion)

        rubros_validos = RUBROS_POR_RECORTE[serie.recorte]
        if rubro is None:
            if len(rubros_validos) != 1:
                raise InvarianteViolado(
                    f"el recorte '{serie.recorte}' de la serie admite varios rubros "
                    f"{sorted(rubros_validos)} — hay que indicar 'rubro' explícito."
                )
            rubro = next(iter(rubros_validos))
        elif rubro not in rubros_validos:
            raise InvarianteViolado(
                f"rubro='{rubro}' no es válido para el recorte '{serie.recorte}' de la serie; "
                f"válidos: {sorted(rubros_validos)}"
            )

        columna_peso = RUBRO_A_COLUMNA_PESO[rubro]
        # dropna() defensivo: hoy (2012/2019/2025) las 7 columnas de peso vienen
        # 100% pobladas (0.0 literal para "no participa"), pero el invariante de
        # CanastaINPP permite NaN estructuralmente — si algún día lo trae, el
        # genérico sale del grupo en vez de reventar en el groupby.
        #
        # Peso EXACTO 0 también sale del grupo, y no es defensivo — es real: con
        # datos 2019 reales, 20 genéricos traen peso 0.0 en "bienes finales" (ej.
        # "085") y esos mismos 20 SÍ faltan en la serie publicada de ese recorte
        # (INEGI no cotiza precio de un genérico que no participa del todo en
        # bienes finales). Exigir que la serie los cubra sería un falso rechazo:
        # peso 0 no aporta nada a Σ(peso·índice)/Σpeso sin importar qué precio
        # (o ausencia de precio) tenga ese genérico.
        ponderador = canasta.df[columna_peso].dropna().astype(float)
        ponderador = ponderador[ponderador != 0]

        if sin_petroleo:
            # Solo excluir del grupo basta: _laspeyres_por_grupo divide entre
            # Σponderador del grupo, que ya no incluye el 070 — la renormalización
            # queda implícita en esa división, sin paso aparte.
            ponderador = ponderador.drop(index=CODIGO_PETROLEO_CRUDO, errors="ignore")

        if ponderador.empty:
            raise CanastaSinGenericos(
                f"la canasta no tiene genéricos utilizables para rubro='{rubro}'"
                + (" con sin_petroleo=True" if sin_petroleo else "")
                + " -- todos los pesos son 0, NaN, o el único genérico con peso era el 070 "
                "(Petróleo crudo), excluido por sin_petroleo."
            )

        nombres: pd.Series | None = None
        if agregacion == TIPO_INPP:
            categoria_por_generico = pd.Series(TIPO_INPP, index=ponderador.index)
        elif agregacion == "MERCANCIAS_SERVICIOS":
            sector = canasta.df.loc[ponderador.index, _COLUMNA_SECTOR]
            categoria_por_generico = sector.isin(SECTORES_MERCANCIAS).map(
                {True: "MERCANCIAS", False: "SERVICIOS"}
            )
        else:
            columna = AGREGACION_A_COLUMNA[agregacion]
            categoria_por_generico = canasta.df.loc[ponderador.index, columna].dropna()
            ponderador = ponderador.loc[categoria_por_generico.index]

            # Nombre legible del nivel, best-effort: la columna sin "codigo "
            # (ej. "rama") trae el texto original — combinado ("1111 nombre...")
            # solo si la canasta se generó con --canasta. Si viene bare (idéntica
            # al código), no hay nombre real que mostrar y se omite del todo
            # (Vista.ancho no agrega columna "nombre" vacía).
            columna_texto = columna.removeprefix("codigo ")
            texto_por_grupo = (
                canasta.df.loc[categoria_por_generico.index, columna_texto]
                .groupby(categoria_por_generico)
                .first()
            )
            if not texto_por_grupo.astype(str).str.fullmatch(_PATRON_BARE).all():
                nombres = texto_por_grupo

        genericos_del_grupo = categoria_por_generico.index
        _validar_serie_cubre_grupo(genericos_del_grupo, serie, canasta.version, agregacion, rubro)

        serie_recortada = _recortar_series_fecha(serie.df.loc[genericos_del_grupo], canasta.version)
        serie_rellenada, df_diagnostico_relleno, _ = _rellenar_dato_serie_faltante(
            serie_recortada, canasta.version, agregacion, rubro
        )

        indice_por_grupo = _laspeyres_por_grupo(serie_rellenada, ponderador, categoria_por_generico)
        indice_incidencia_por_grupo = indice_por_grupo.copy()  # antes de un eventual rebase futuro

        # NaN irrellenable → sin_datos; celda imputada → rellenado
        hay_faltante = serie_rellenada.isna().groupby(categoria_por_generico).any()
        hubo_relleno = (
            (serie_recortada.isna() & serie_rellenada.notna()).groupby(categoria_por_generico).any()
        )

        indice_apilado = indice_por_grupo.T.stack()
        indice_apilado.index = indice_apilado.index.set_names(["periodo", "indice"])
        idx = indice_apilado.index

        faltante_alineado = hay_faltante.T.stack().reindex(idx).fillna(False)
        relleno_alineado = hubo_relleno.T.stack().reindex(idx).fillna(False)

        faltante_bool = faltante_alineado.to_numpy(dtype=bool)
        relleno_bool = relleno_alineado.to_numpy(dtype=bool)
        estado_calculo_arr = np.where(
            faltante_bool, "sin_datos", np.where(relleno_bool & ~faltante_bool, "rellenado", "ok")
        )
        motivo_error_arr = np.where(faltante_bool, "faltantes en serie", None)  # type: ignore[call-overload]

        df_indice_resultado = pd.DataFrame(
            {
                "version": canasta.version,
                "agregacion": agregacion,
                "rubro": rubro,
                "indice_replicado": indice_apilado.where(~faltante_bool).values,
                "indice_incidencia": (
                    indice_incidencia_por_grupo.T.stack().reindex(idx).where(~faltante_bool).values
                ),
                "estado_calculo": estado_calculo_arr,
                "motivo_error": motivo_error_arr,
            },
            index=idx,
        )

        cubierto = serie_rellenada.notna()
        con_indice_por_grupo = cubierto.groupby(categoria_por_generico).sum()
        ponderador_cubierto_por_grupo = (
            cubierto.multiply(ponderador, axis=0).groupby(categoria_por_generico).sum()
        )
        genericos_esperados_por_grupo = categoria_por_generico.groupby(
            categoria_por_generico
        ).size()
        ponderador_esperado_por_grupo = ponderador.groupby(categoria_por_generico).sum()

        con_indice_flat = con_indice_por_grupo.T.stack().reindex(idx).to_numpy()
        ponderador_cubierto_flat = ponderador_cubierto_por_grupo.T.stack().reindex(idx).to_numpy()
        grupos_col = idx.get_level_values("indice")
        genericos_esperados_flat = genericos_esperados_por_grupo.reindex(grupos_col).to_numpy()
        ponderador_esperado_flat = ponderador_esperado_por_grupo.reindex(grupos_col).to_numpy()

        df_reporte = pd.DataFrame(
            {
                "version": canasta.version,
                "estado_calculo": estado_calculo_arr,
                "motivo_error": motivo_error_arr,
                "genericos_esperados": genericos_esperados_flat,
                "genericos_con_indice": con_indice_flat,
                "genericos_sin_indice": genericos_esperados_flat - con_indice_flat,
                "cobertura_genericos_pct": 100.0 * con_indice_flat / genericos_esperados_flat,
                "ponderador_esperado": ponderador_esperado_flat,
                "ponderador_cubierto": ponderador_cubierto_flat,
            },
            index=idx,
        )

        df_diagnostico = pd.concat(
            [
                _construir_diagnostico(serie_rellenada, canasta.version, agregacion, rubro),
                df_diagnostico_relleno,
            ],
            ignore_index=True,
        )

        manifiesto = ManifestCalculo(
            version=canasta.version,
            agregacion=agregacion,
            rubro=rubro,
            sin_petroleo=sin_petroleo,
            calculador="LaspeyresDirecto",
            ruta_canasta=ruta_canasta,
            ruta_series=ruta_serie,
            fecha=fecha,
        )
        return ResultadoIndice(
            df_indice_resultado, [manifiesto], df_reporte, df_diagnostico, nombres=nombres
        )
