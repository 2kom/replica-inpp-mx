from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

from replica_inpp.dominio.calculo.base import (
    _COLUMNAS_DIAGNOSTICO,
    CalculadorBase,
    _construir_diagnostico,
    _laspeyres_por_grupo,
    _recortar_series_fecha,
    _rellenar_dato_serie_faltante,
    _validar_serie_cubre_grupo,
)
from replica_inpp.dominio.errores import (
    CanastaSinGenericos,
    ErrorCalculo,
    InvarianteViolado,
    VersionNoCoincide,
)
from replica_inpp.dominio.modelos.canasta import CanastaINPP
from replica_inpp.dominio.modelos.indice import ResultadoIndice
from replica_inpp.dominio.modelos.serie import SerieNormalizada
from replica_inpp.dominio.tipos import (
    AGREGACION_A_COLUMNA,
    AGREGACION_A_NOMBRE,
    AGREGACIONES_VALIDAS,
    CODIGO_PETROLEO_CRUDO,
    COLUMNA_ENCADENAMIENTO_POR_RECORTE,
    RANGOS_CANASTAS,
    RUBRO_A_COLUMNA_PESO,
    RUBROS_POR_RECORTE,
    SECTORES_MERCANCIAS,
    TIPO_INPP,
    ManifestCalculo,
)

_COLUMNA_SECTOR = "codigo sector"

# Texto bare ("1111", igual al código, sin nombre real) vs combinado ("1111
# Cultivo de cereales..."). Ver `LaspeyresDirecto` / `LectorCanastaCsv`.
_PATRON_BARE = re.compile(r"^\d+$")


class LaspeyresEncadenado(CalculadorBase):
    """Etapa 3 (encadenamiento 2025) — solo aplica a `canasta.version == 2025`.

    Único empalme de canasta que existe hasta ahora (2019→2025); a diferencia
    de `replica-inpc-mx` (varios empalmes históricos, `LaspeyresEncadenadoT1`/
    `LaspeyresEncadenadoT2`) acá basta una sola clase.

    La serie cruda (`SerieNormalizada`) viene en escala absoluta continua, no
    en base local ~100 (jul-2025 en `s25` vale ~107-188 — CLAUDE.md, sección
    "Base"). El factor de encadenamiento por genérico (`f_j`, columna de
    `CanastaINPP` según el `recorte` de la serie — ver
    `COLUMNA_ENCADENAMIENTO_POR_RECORTE`, ya resuelto por INEGI en
    `factor_de_encadenamiento_ti.xlsx`, fusiones/desagregaciones/altas de la
    reclasificación SCIAN 2013→2018 incluidas) sirve para:

    1. De-encadenar la serie cruda a escala local de la canasta 2025:
       `df_base = serie / f_j`.
    2. Agregar `df_base` con las ponderaciones 2025 (`_laspeyres_por_grupo`)
       → `i_tramo`, el índice en su propia escala local.
    3. Encadenar de vuelta a escala absoluta con `factor_h = I_viejo(traslape)
       / 100` — el índice de la MISMA `agregacion`/`rubro`/`incluir_petroleo`,
       tomado de `referencia` (el `ResultadoIndice` de esa combinación con
       `canasta.version=2019`, ya calculado con `LaspeyresDirecto` — p.ej. vía
       `api/indices.py::calcular_indice`), evaluado en el periodo de
       traslape. No es el promedio ponderado de los `f_j` hijos con el peso
       2025 (fórmula de `LaspeyresEncadenadoT2` en `replica-inpc-mx`): ese
       promedio usa ponderador NUEVO donde el manual pide "ponderaciones a
       julio de 2019" — ponderador VIEJO. `referencia` ya trae ese valor
       directo, sin reconciliar códigos de genérico entre versiones (dos
       cálculos independientes, cada uno dentro de su propia clasificación).
       Para un grupo que no existía en 2019 (cero genéricos, ej. una rama SCIAN
       nueva tras la reclasificación 2013→2018 — no hay fila que agregar, no
       "existía pero se perdió") no hay ponderador viejo que usar: ahí sí se
       cae al promedio ponderado con peso 2025 de `f_j` (la fórmula de
       `LaspeyresEncadenadoT2`, con el peso normalizado dentro del grupo antes
       de multiplicar por `f_j` para no desbordar con un `f_j` extremo), solo
       como respaldo para ese caso. Un grupo que SÍ existía en 2019 pero cuyo
       `indice_replicado` en `referencia` es NaN ahí (referencia inválida, ej.
       `estado_calculo="sin_datos"`) no cae a este respaldo — es un error
       distinto, no una categoría nueva.
    4. `resultado = i_tramo · factor_h`.

    Igual que `LaspeyresDirecto`, agrupa por 2 ejes independientes
    (`agregacion` x `rubro`); una corrida cubre una sola combinación.

    A diferencia de `LaspeyresEncadenadoT1`/`T2` de `replica-inpc-mx` (que
    reciben `referencia_empalme_por_indice: dict[str, float]`, ya reducido a
    un valor por categoría), acá `referencia` es el `ResultadoIndice`
    completo tal cual devuelve `LaspeyresDirecto` — sin paso de normalización
    aparte porque `SECTOR`/`INPP`/`MERCANCIAS_SERVICIOS` usan código estable
    entre 2019 y 2025 (no hace falta reconciliar nombre/código de categoría
    entre versiones, a diferencia de INPC).

    Args:
        referencia: `ResultadoIndice` de la MISMA `agregacion`/`rubro`/
            `incluir_petroleo` que se le pedirá a `calcular`, calculado con
            `canasta.version=2019` (típicamente `LaspeyresDirecto` vía
            `calcular_indice`) y que cubra el periodo de traslape (jul-2025).
            Se valida recién en `calcular`, porque `agregacion`/`rubro`/
            `incluir_petroleo` no se conocen hasta esa llamada.

    Raises:
        InvarianteViolado: al calcular, si `canasta.version` no es 2025, si
            `agregacion` no es válida, si `rubro` no es válido para el
            `recorte` de `serie` (o es ambiguo y no se indicó), o si
            `referencia` no tiene un manifiesto con `version=2019` y la
            MISMA `agregacion`/`rubro`/`incluir_petroleo` que se está pidiendo.
        VersionNoCoincide: `serie` trae metadata de versión (`cargar_serie`) y
            no coincide con `canasta.version`.
        CanastaSinGenericos: tras filtrar pesos NaN/0 (y el 070 si
            `incluir_petroleo=False`), no queda ningún genérico utilizable.
        ErrorCalculo: a `serie` le faltan genéricos que el grupo necesita; la
            columna de encadenamiento del `recorte` está totalmente vacía
            (canasta generada sin `--encadenamientos`); `referencia` no
            cubre el periodo de traslape; algún grupo existe en `referencia`
            pero su `indice_replicado` es NaN ahí (referencia inválida, no
            categoría nueva — no cae a respaldo); o algún grupo de
            `agregacion` no tiene `factor_h` ni en el tramo anterior ni por
            respaldo (no todos sus genéricos tienen `f_j` válido).
    """

    def __init__(self, referencia: ResultadoIndice) -> None:
        self._referencia = referencia

    def calcular(
        self,
        canasta: CanastaINPP,
        serie: SerieNormalizada,
        agregacion: str,
        rubro: str | None = None,
        *,
        incluir_petroleo: bool = True,
    ) -> ResultadoIndice:
        fecha = datetime.now()
        ruta_canasta = canasta.df.attrs.get("origen")
        ruta_serie = serie.df.attrs.get("origen")

        if canasta.version != 2025:
            raise InvarianteViolado(
                f"LaspeyresEncadenado requiere canasta.version=2025; recibió {canasta.version}"
            )

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
        ponderador = canasta.df[columna_peso].dropna().astype(float)
        ponderador = ponderador[ponderador != 0]

        if not incluir_petroleo:
            ponderador = ponderador.drop(index=CODIGO_PETROLEO_CRUDO, errors="ignore")

        if ponderador.empty:
            raise CanastaSinGenericos(
                f"la canasta no tiene genéricos utilizables para rubro='{rubro}'"
                + (" con incluir_petroleo=False" if not incluir_petroleo else "")
                + " -- todos los pesos son 0, NaN, o el único genérico con peso era el 070 "
                "(Petróleo crudo), excluido por incluir_petroleo=False."
            )

        # f_j sale del RECORTE de la serie, no del rubro -- varios rubros comparten
        # recorte (ej. demanda_interna_total/consumo/capital, los 3 sobre
        # mercado_nacional) y por tanto la misma columna de encadenamiento.
        columna_encadenamiento = COLUMNA_ENCADENAMIENTO_POR_RECORTE[serie.recorte]
        f_j_total = canasta.df[columna_encadenamiento].astype(float)
        if f_j_total.dropna().empty:
            raise ErrorCalculo(
                f"La canasta versión {canasta.version} no trae factor de encadenamiento en "
                f"'{columna_encadenamiento}' -- se generó sin --encadenamientos."
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

        f_j = f_j_total.loc[genericos_del_grupo]

        serie_recortada = _recortar_series_fecha(serie.df.loc[genericos_del_grupo], canasta.version)
        serie_rellenada, df_diagnostico_relleno, _ = _rellenar_dato_serie_faltante(
            serie_recortada, canasta.version, agregacion, rubro
        )

        # De-encadena a escala local (~100 en jul-2025 para el genérico que empalma
        # limpio). `f_j` NaN es real -- INEGI marca "N/A" por genérico en
        # exportación/uso final (ver `CanastaINPP._COLUMNAS_ENCADENAMIENTO_PARCIAL`)
        # -- se propaga como hueco de dato, mismo tratamiento que un faltante de
        # serie (columna entera de ese genérico queda NaN, ver `hay_faltante` abajo).
        df_base = serie_rellenada.divide(f_j, axis=0)

        i_tramo = _laspeyres_por_grupo(df_base, ponderador, categoria_por_generico)

        # factor_h = I_viejo(traslape)/100, MISMO agregacion/rubro/incluir_petroleo pero
        # con canasta.version=2019 -- ver docstring de la clase. `referencia` ya
        # viene calculado (no se recalcula acá), nunca cruza código de genérico
        # entre versiones.
        manifiestos_referencia = [
            m
            for m in self._referencia.manifiesto
            if m.version == 2019
            and m.agregacion == agregacion
            and m.rubro == rubro
            and m.incluir_petroleo == incluir_petroleo
        ]
        if len(manifiestos_referencia) != 1:
            raise InvarianteViolado(
                f"referencia no tiene un manifiesto único con version=2019, agregacion="
                f"'{agregacion}', rubro='{rubro}', incluir_petroleo={incluir_petroleo} -- tiene "
                f"{len(manifiestos_referencia)}. referencia debe venir de calcular_indice/"
                "LaspeyresDirecto con la MISMA combinación que se pide acá."
            )

        # Filtra ANTES de pivotear: `referencia` puede traer más de un manifiesto
        # (p.ej. otra version/agregacion/rubro, cada uno con sus propias filas,
        # válido por el invariante de ResultadoIndice) -- sin este filtro, `.ancho`
        # apila filas de TODOS los manifiestos bajo la misma etiqueta de `indice`, y
        # un periodo que solo existe en un manifiesto ajeno se coló como si fuera
        # del manifiesto que ya validamos arriba (regresión real, confirmada con un
        # `ResultadoIndice` de 2 manifiestos por índices públicos, sin mutar nada
        # privado -- ver test_referencia_con_manifiesto_ajeno_en_traslape_lanza_error_calculo).
        traslape = RANGOS_CANASTAS[canasta.version][0]
        largo_referencia = self._referencia.resultado.largo
        filtro_manifiesto = (
            (largo_referencia["version"] == 2019)
            & (largo_referencia["agregacion"] == agregacion)
            & (largo_referencia["rubro"] == rubro)
        )
        ancho_anterior = largo_referencia.loc[filtro_manifiesto, "indice_replicado"].unstack(
            "periodo"
        )
        if traslape not in ancho_anterior.columns:
            raise ErrorCalculo(
                f"referencia no tiene el periodo de traslape {traslape} dentro del manifiesto "
                f"version=2019/agregacion='{agregacion}'/rubro='{rubro}' -- no se puede "
                "calcular factor_h."
            )
        factor_h_total = ancho_anterior[traslape].astype(float) / 100
        grupos_del_resultado = pd.Index(i_tramo.index)

        # Grupo AUSENTE de `factor_h_total.index` (nunca existió en 2019, ej. una
        # rama SCIAN nueva tras la reclasificación 2013→2018) vs grupo PRESENTE
        # pero con `indice_replicado` NaN en el traslape (existía en 2019, pero
        # su referencia quedó sin dato ahí, típicamente `estado_calculo=
        # "sin_datos"`) -- son fallas distintas. Solo el primero cae a respaldo;
        # el segundo es una referencia inválida, no una categoría nueva, y
        # tratarlo igual que una ausencia real sustituiría en silencio un dato
        # que debería fallar por el respaldo de un grupo que en realidad sí
        # tiene ponderador viejo (nunca debe usarse peso 2025 ahí).
        grupos_con_referencia_invalida = [
            grupo
            for grupo in grupos_del_resultado
            if grupo in factor_h_total.index and pd.isna(factor_h_total.loc[grupo])
        ]
        if grupos_con_referencia_invalida:
            raise ErrorCalculo(
                f"referencia tiene 'indice_replicado' en NaN para {grupos_con_referencia_invalida} "
                f"en el traslape {traslape} -- esos grupos existían en 2019 pero su referencia "
                "quedó sin dato ahí (probable estado_calculo='sin_datos'), no es una categoría "
                "nueva y no se puede encadenar con una referencia inválida."
            )

        factor_h = factor_h_total.reindex(grupos_del_resultado)
        grupos_sin_referencia = grupos_del_resultado[factor_h.isna()]
        if len(grupos_sin_referencia) > 0:
            # Categoría nueva en 2025 (cero genéricos en 2019) -- no hay
            # "ponderador de julio 2019" que usar porque el grupo no existía, así
            # que se cae a respaldo: promedio ponderado (peso 2025) del propio
            # f_j de sus genéricos, mismo mecanismo que usa por defecto
            # LaspeyresEncadenadoT2 de replica-inpc-mx. Solo de respaldo -- para
            # los grupos que SÍ tienen referencia, el método de arriba sigue
            # siendo más fiel al manual (usa ponderador viejo, no el nuevo).
            # Un grupo con ALGÚN genérico sin f_j válido no cuenta como respaldo
            # confiable -- promediar solo los genéricos que sí tienen f_j daría
            # un número silenciosamente sesgado (divide entre el ponderador
            # completo del grupo, no solo el de los genéricos con dato). Se
            # excluye el grupo entero de `factor_h_respaldo` (queda NaN, `sum`
            # normal sobre un grupo sin ninguna fila da 0.0 por defecto en
            # pandas, no NaN -- `min_count=1` evita ese silencio).
            #
            # Peso NORMALIZADO (peso/suma_del_grupo, ≤1) antes de multiplicar por
            # f_j, no peso crudo -- `peso_crudo · f_j` puede desbordar a inf antes
            # de dividir (ej. peso=40, f_j=1e307 → 4e308 > float64 max) aunque el
            # promedio ponderado final sea perfectamente finito (~1e307).
            # Normalizar primero acota cada término a ≤ f_j, sin cambiar el
            # resultado en el rango normal de `f_j` (~1.0-2.0 en datos reales).
            grupo_completo = f_j.notna().groupby(categoria_por_generico).transform("all")
            peso_normalizado = ponderador / ponderador.groupby(categoria_por_generico).transform(
                "sum"
            )
            factor_h_respaldo: pd.Series = (
                (peso_normalizado * f_j)
                .where(grupo_completo)
                .groupby(categoria_por_generico)
                .sum(min_count=1)
            )
            factor_h = factor_h.fillna(factor_h_respaldo.reindex(grupos_del_resultado))

        grupos_sin_factor = grupos_del_resultado[factor_h.isna()].tolist()
        if grupos_sin_factor:
            raise ErrorCalculo(
                f"No hay factor_h para {grupos_sin_factor} en el traslape {traslape} -- no están "
                "en el tramo anterior (referencia) y el respaldo no se pudo calcular porque no "
                "todos los genéricos de estos grupos tienen un factor de encadenamiento válido."
            )

        resultado_por_grupo = i_tramo.multiply(factor_h, axis=0)

        # i_tramo y factor_h ya vienen finitos por separado (cada uno pasó por su
        # propio _laspeyres_por_grupo), pero el producto de dos valores finitos
        # puede desbordar a inf -- validar de nuevo acá, mismo criterio que
        # _laspeyres_por_grupo.
        valores_resultado = resultado_por_grupo.to_numpy(dtype=float)
        if (~pd.isna(valores_resultado) & ~np.isfinite(valores_resultado)).any():
            raise ErrorCalculo(
                "El encadenamiento produjo un valor no finito -- posible desbordamiento al "
                "multiplicar i_tramo por factor_h."
            )

        indice_incidencia_por_grupo = resultado_por_grupo.copy()

        # Cobertura sobre `df_base` (post de-encadenamiento), no sobre la serie
        # cruda -- un genérico con precio pero `f_j` NaN igual queda sin dato
        # utilizable para este cálculo. `serie_faltante`/`fj_faltante_por_celda` se
        # separan para no reportar "faltantes en serie" cuando el hueco real es de
        # encadenamiento (INEGI marca "N/A" por genérico en exportación/uso final).
        serie_faltante = serie_rellenada.isna()
        fj_faltante_por_celda = pd.DataFrame(
            {col: f_j.isna() for col in serie_rellenada.columns}, index=serie_rellenada.index
        )
        hay_faltante = df_base.isna().groupby(categoria_por_generico).any()
        hay_fj_faltante = fj_faltante_por_celda.groupby(categoria_por_generico).any()
        hay_serie_faltante = serie_faltante.groupby(categoria_por_generico).any()
        hubo_relleno = (
            (serie_recortada.isna() & serie_rellenada.notna()).groupby(categoria_por_generico).any()
        )

        resultado_apilado = resultado_por_grupo.T.stack()
        resultado_apilado.index = resultado_apilado.index.set_names(["periodo", "indice"])
        idx = resultado_apilado.index

        faltante_alineado = hay_faltante.T.stack().reindex(idx).fillna(False)
        fj_alineado = hay_fj_faltante.T.stack().reindex(idx).fillna(False)
        serie_alineado = hay_serie_faltante.T.stack().reindex(idx).fillna(False)
        relleno_alineado = hubo_relleno.T.stack().reindex(idx).fillna(False)

        faltante_bool = faltante_alineado.to_numpy(dtype=bool)
        fj_bool = fj_alineado.to_numpy(dtype=bool)
        serie_bool = serie_alineado.to_numpy(dtype=bool)
        relleno_bool = relleno_alineado.to_numpy(dtype=bool)
        estado_calculo_arr = np.where(
            faltante_bool, "sin_datos", np.where(relleno_bool & ~faltante_bool, "rellenado", "ok")
        )
        # Prioridad f_j sobre serie cuando ambas causas coinciden en el mismo grupo:
        # un genérico con f_j ausente queda sin dato en TODOS sus periodos sin
        # importar el precio, es la causa estructural.
        motivo_fj = f"factor de encadenamiento ausente en '{columna_encadenamiento}'"
        motivo_error_arr = np.select(
            [faltante_bool & fj_bool, faltante_bool & serie_bool],
            [motivo_fj, "faltantes en serie"],
            default=None,  # type: ignore[arg-type]
        )

        df_indice_resultado = pd.DataFrame(
            {
                "version": canasta.version,
                "agregacion": agregacion,
                "rubro": rubro,
                "indice_replicado": resultado_apilado.where(~faltante_bool).values,
                "indice_incidencia": (
                    indice_incidencia_por_grupo.T.stack().reindex(idx).where(~faltante_bool).values
                ),
                "estado_calculo": estado_calculo_arr,
                "motivo_error": motivo_error_arr,
            },
            index=idx,
        )

        cubierto = df_base.notna()
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

        # Diagnóstico de f_j ausente aparte del de serie faltante: un genérico con
        # f_j NaN queda excluido de `_construir_diagnostico` sobre `df_base` (que
        # reportaría "valor NaN en serie publicada", causa equivocada) y se reporta
        # con su propio detalle en su lugar.
        generos_fj_faltante = f_j.index[f_j.isna()]
        df_diagnostico_serie = _construir_diagnostico(
            df_base.drop(index=generos_fj_faltante, errors="ignore"),
            canasta.version,
            agregacion,
            rubro,
        )
        if len(generos_fj_faltante) > 0:
            df_diagnostico_fj = pd.DataFrame(
                [
                    {
                        "version": canasta.version,
                        "agregacion": agregacion,
                        "rubro": rubro,
                        "periodo": periodo,
                        "generico": generico,
                        "nivel_faltante": "periodo",
                        "tipo_faltante": "factor_encadenamiento",
                        "detalle": motivo_fj,
                    }
                    for generico in generos_fj_faltante
                    for periodo in df_base.columns
                ],
                columns=_COLUMNAS_DIAGNOSTICO,
            )
        else:
            df_diagnostico_fj = pd.DataFrame(columns=_COLUMNAS_DIAGNOSTICO)

        df_diagnostico = pd.concat(
            [df_diagnostico_serie, df_diagnostico_fj, df_diagnostico_relleno],
            ignore_index=True,
        )

        manifiesto = ManifestCalculo(
            version=canasta.version,
            agregacion=agregacion,
            rubro=rubro,
            incluir_petroleo=incluir_petroleo,
            calculador="LaspeyresEncadenado",
            ruta_canasta=ruta_canasta,
            ruta_series=ruta_serie,
            fecha=fecha,
        )
        return ResultadoIndice(
            df_indice_resultado, [manifiesto], df_reporte, df_diagnostico, nombres=nombres
        )
