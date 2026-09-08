from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from replica_inpp.api import config, consultas
from replica_inpp.dominio.errores import ErrorConfiguracion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.infraestructura.inegi.fuente_validacion_api import FuenteValidacionApi

# Sector de 1 sola variante publicada (ver `_SECTORES`) — usado para los casos
# donde `incluir_petroleo` no aplica.
_TIPO = "11"
# Único tipo con 2 variantes publicadas (con/sin petróleo) hoy.
_TIPO_CON_VARIANTE = "21"


@pytest.fixture(autouse=True)
def _config_valida(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INEGI_TOKEN", raising=False)
    config._token = None
    config.set_token("tok")
    config.timeout_api = 10
    FuenteValidacionApi._cache.clear()
    yield
    config._token = None
    config.timeout_api = 10
    FuenteValidacionApi._cache.clear()


def _mock_resp(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


_RESPUESTA_MENSUAL = {
    "Series": [
        {
            "OBSERVATIONS": [
                {"TIME_PERIOD": "2026/03", "OBS_VALUE": "145.200", "OBS_STATUS": "3"},
                {"TIME_PERIOD": "2026/02", "OBS_VALUE": "144.300", "OBS_STATUS": "3"},
            ]
        }
    ]
}


def _respuesta_con_valor(valor: float) -> dict:
    return {"Series": [{"OBSERVATIONS": [{"TIME_PERIOD": "2026/03", "OBS_VALUE": str(valor)}]}]}


def _side_effect_valor_por_indicador(mapa: dict[str, float]):
    """`side_effect` para `requests.get`: cada URL responde con el valor que le
    corresponde a SU indicador según `mapa` (buscado por `/INDICATOR/<id>/` en
    la URL) — prueba, por el valor devuelto, que llegó el ID BIE correcto a la
    red, no solo que la columna resultante se llame como se esperaba. Un
    indicador fuera de `mapa` hace explotar el mock con un mensaje que nombra
    la URL real, en vez de devolver un valor cualquiera en silencio.
    """

    def _side_effect(url: str, timeout: float | None = None) -> MagicMock:
        for indicador, valor in mapa.items():
            if f"/INDICATOR/{indicador}/" in url:
                return _mock_resp(_respuesta_con_valor(valor))
        raise AssertionError(f"requests.get llamado con indicador inesperado: {url}")

    return _side_effect


# -- timeout inválido ------------------------------------------------------------


def test_timeout_invalido_lanza_error_configuracion_sin_tocar_red(mocker) -> None:
    mock_get = mocker.patch("requests.get")
    config.timeout_api = 0

    with pytest.raises(ErrorConfiguracion, match="timeout"):
        consultas.consultar_indice(_TIPO)

    assert mock_get.call_count == 0


# -- consultar_indice: ruta exitosa, tipo de 1 sola variante ---------------------


def test_consultar_indice_devuelve_dataframe_de_una_columna(mocker) -> None:
    mocker.patch("requests.get", return_value=_mock_resp(_RESPUESTA_MENSUAL))

    df = consultas.consultar_indice(_TIPO)

    assert df.index.name == "periodo"
    assert list(df.index) == [PeriodoMensual(2026, 2), PeriodoMensual(2026, 3)]  # ordenado
    assert list(df.columns) == [_TIPO]  # 1 sola columna, nombrada como el tipo pedido
    # float("145.200") directo del parseo, sin cálculo de por medio — igualdad
    # exacta, no tolerancia (145.2001 pasaría el default de approx).
    assert df[_TIPO][PeriodoMensual(2026, 3)] == 145.200  # type: ignore[call-overload]


def test_consultar_indice_pide_el_id_correcto_a_la_red(mocker) -> None:
    # "1700004" es el único ID real de _SECTORES["11"] — cualquier otro ID que
    # requests.get reciba hace explotar el side_effect (ver su docstring).
    mocker.patch("requests.get", side_effect=_side_effect_valor_por_indicador({"1700004": 11.0}))

    df = consultas.consultar_indice(_TIPO)

    assert df[_TIPO][PeriodoMensual(2026, 3)] == 11.0  # type: ignore[call-overload]


def test_consultar_indice_tipo_invalido_lanza_error_configuracion() -> None:
    with pytest.raises(ErrorConfiguracion):
        consultas.consultar_indice("tipo_inexistente")


# -- incluir_petroleo: reglas de catálogo ----------------------------------------


def test_incluir_petroleo_explicito_en_tipo_de_1_variante_lanza_error(mocker) -> None:
    mock_get = mocker.patch("requests.get")

    with pytest.raises(ErrorConfiguracion, match="no tiene 2 variantes"):
        consultas.consultar_indice(_TIPO, incluir_petroleo=False)

    assert mock_get.call_count == 0  # falla antes de tocar red


def test_incluir_petroleo_none_en_tipo_de_2_variantes_lanza_error_ambiguo(mocker) -> None:
    mock_get = mocker.patch("requests.get")

    with pytest.raises(ErrorConfiguracion, match="2 variantes publicadas"):
        consultas.consultar_indice(_TIPO_CON_VARIANTE)

    assert mock_get.call_count == 0


def test_incluir_petroleo_true_false_en_tipo_de_2_variantes_resuelve(mocker) -> None:
    # Valores DISTINTOS por ID ("1700162"=sin, "1700163"=con) — si el código
    # invirtiera las 2 claves de `_SECTORES["21"]`, este test lo detectaría
    # (a diferencia de comparar solo `.columns`, que da igual con cualquier ID).
    mocker.patch(
        "requests.get",
        side_effect=_side_effect_valor_por_indicador({"1700162": 162.0, "1700163": 163.0}),
    )

    df_sin = consultas.consultar_indice(_TIPO_CON_VARIANTE, incluir_petroleo=False)
    FuenteValidacionApi._cache.clear()
    df_con = consultas.consultar_indice(_TIPO_CON_VARIANTE, incluir_petroleo=True)

    assert df_sin[_TIPO_CON_VARIANTE][PeriodoMensual(2026, 3)] == 162.0  # type: ignore[call-overload]
    assert df_con[_TIPO_CON_VARIANTE][PeriodoMensual(2026, 3)] == 163.0  # type: ignore[call-overload]


# -- consultar_indice: eje rubro (_RUBROS) — "INPP" + destino/etapa ---------------


def test_consultar_indice_inpp_con_sin_petroleo_resuelve(mocker) -> None:
    # "1700001"=sin petróleo, "1700002"=con — mismo criterio que sector 21.
    mocker.patch(
        "requests.get",
        side_effect=_side_effect_valor_por_indicador({"1700001": 1.0, "1700002": 2.0}),
    )

    df_sin = consultas.consultar_indice("INPP", incluir_petroleo=False)
    FuenteValidacionApi._cache.clear()
    df_con = consultas.consultar_indice("INPP", incluir_petroleo=True)

    assert df_sin["INPP"][PeriodoMensual(2026, 3)] == 1.0  # type: ignore[call-overload]
    assert df_con["INPP"][PeriodoMensual(2026, 3)] == 2.0  # type: ignore[call-overload]


def test_consultar_indice_inpp_none_es_ambiguo(mocker) -> None:
    mock_get = mocker.patch("requests.get")

    with pytest.raises(ErrorConfiguracion, match="2 variantes publicadas"):
        consultas.consultar_indice("INPP")

    assert mock_get.call_count == 0


@pytest.mark.parametrize(
    ("tipo", "indicador"),
    [
        ("bienes_intermedios", "1750002"),
        ("demanda_interna_total", "1380015"),
        ("demanda_interna_consumo", "1380016"),
        ("demanda_interna_capital", "1380017"),
        ("exportaciones", "1380018"),
    ],
)
def test_consultar_indice_rubros_de_1_variante_resuelven(mocker, tipo, indicador) -> None:
    mocker.patch("requests.get", side_effect=_side_effect_valor_por_indicador({indicador: 1.0}))

    df = consultas.consultar_indice(tipo)

    assert df[tipo][PeriodoMensual(2026, 3)] == 1.0  # type: ignore[call-overload]


def test_consultar_indice_bienes_finales_no_soportado(mocker) -> None:
    # Sin ID BIE real conocido — probado exhaustivo contra las 23 series de
    # nivel disponibles, ninguna calza (ver `_RUBROS` en fuente_validacion_api.py).
    mock_get = mocker.patch("requests.get")

    with pytest.raises(ErrorConfiguracion, match="no tiene indicador INEGI disponible"):
        consultas.consultar_indice("bienes_finales")

    assert mock_get.call_count == 0


# -- consultar_variacion: mismo mecanismo, catálogo distinto (incluye "INPP") ----
# frecuencia es posicional obligatoria, sin default — "mensual"/"anual" tienen
# catálogo, "acumulada" todavía no.


def test_consultar_variacion_devuelve_dataframe_de_una_columna(mocker) -> None:
    mocker.patch("requests.get", return_value=_mock_resp(_RESPUESTA_MENSUAL))

    df = consultas.consultar_variacion(_TIPO, "mensual")

    assert df.index.name == "periodo"
    assert list(df.columns) == [_TIPO]
    assert df[_TIPO][PeriodoMensual(2026, 3)] == 145.200  # type: ignore[call-overload]


def test_consultar_variacion_mensual_pide_el_id_correcto_a_la_red(mocker) -> None:
    # "1800003" es el único ID de "11" en `_VARIACION_MENSUAL` — un mock que
    # solo mapea ese ID hace explotar el side_effect si el código consultara
    # por error el de otra frecuencia (ej. "1801003" de interanual).
    mocker.patch("requests.get", side_effect=_side_effect_valor_por_indicador({"1800003": 3.0}))

    df = consultas.consultar_variacion(_TIPO, "mensual")

    assert df[_TIPO][PeriodoMensual(2026, 3)] == 3.0  # type: ignore[call-overload]


@pytest.mark.parametrize("frecuencia", ["mensual", "anual", "acumulada"])
def test_consultar_variacion_tipo_invalido_lanza_error_configuracion(frecuencia) -> None:
    with pytest.raises(ErrorConfiguracion):
        consultas.consultar_variacion("tipo_inexistente", frecuencia)


def test_consultar_variacion_frecuencia_invalida_lanza_error(mocker) -> None:
    mock_get = mocker.patch("requests.get")

    with pytest.raises(ErrorConfiguracion, match="no válida"):
        consultas.consultar_variacion(_TIPO, "trimestral")  # type: ignore[arg-type]

    assert mock_get.call_count == 0


def test_consultar_variacion_acumulada_resuelve(mocker) -> None:
    # "1802003" es el ID de "11" en `_VARIACION_ACUMULADA` — distinto del de
    # mensual ("1800003") y anual ("1801003"), ver docstring del side_effect.
    mocker.patch("requests.get", side_effect=_side_effect_valor_por_indicador({"1802003": 3.0}))

    df = consultas.consultar_variacion(_TIPO, "acumulada")

    assert df[_TIPO][PeriodoMensual(2026, 3)] == 3.0  # type: ignore[call-overload]


def test_consultar_variacion_inpp_requiere_incluir_petroleo(mocker) -> None:
    mock_get = mocker.patch("requests.get")

    with pytest.raises(ErrorConfiguracion, match="2 variantes publicadas"):
        consultas.consultar_variacion("INPP", "mensual")

    assert mock_get.call_count == 0


def test_consultar_variacion_inpp_con_incluir_petroleo_resuelve(mocker) -> None:
    # "1800001"=sin petróleo, "1800002"=con — mismo criterio que el test de
    # sector 21: valores distintos por ID, no solo nombre de columna.
    mocker.patch(
        "requests.get",
        side_effect=_side_effect_valor_por_indicador({"1800001": 1.0, "1800002": 2.0}),
    )

    df_sin = consultas.consultar_variacion("INPP", "mensual", incluir_petroleo=False)
    FuenteValidacionApi._cache.clear()
    df_con = consultas.consultar_variacion("INPP", "mensual", incluir_petroleo=True)

    assert df_sin["INPP"][PeriodoMensual(2026, 3)] == 1.0  # type: ignore[call-overload]
    assert df_con["INPP"][PeriodoMensual(2026, 3)] == 2.0  # type: ignore[call-overload]


def test_consultar_variacion_anual_resuelve(mocker) -> None:
    # "1801003" es el ID de "11" en `_VARIACION_INTERANUAL` — distinto del de
    # mensual ("1800003") y acumulada ("1802003").
    mocker.patch("requests.get", side_effect=_side_effect_valor_por_indicador({"1801003": 3.0}))

    df = consultas.consultar_variacion(_TIPO, "anual")

    assert df[_TIPO][PeriodoMensual(2026, 3)] == 3.0  # type: ignore[call-overload]
