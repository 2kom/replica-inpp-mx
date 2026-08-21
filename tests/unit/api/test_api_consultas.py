from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from replica_inpp.api import config, consultas
from replica_inpp.dominio.errores import ErrorConfiguracion
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.infraestructura.inegi.fuente_validacion_api import FuenteValidacionApi

_TIPO = "PRODUCCION TOTAL"


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


# -- timeout inválido ------------------------------------------------------------


def test_timeout_invalido_lanza_error_configuracion_sin_tocar_red(mocker) -> None:
    mock_get = mocker.patch("requests.get")
    config.timeout_api = 0

    with pytest.raises(ErrorConfiguracion, match="timeout"):
        consultas.consultar_indice(_TIPO)

    assert mock_get.call_count == 0


# -- consultar_indice: ruta exitosa ---------------------------------------------


def test_consultar_indice_devuelve_dataframe(mocker) -> None:
    mocker.patch("requests.get", return_value=_mock_resp(_RESPUESTA_MENSUAL))

    df = consultas.consultar_indice("produccion total")  # minúsculas: se normaliza con .upper()

    assert df.index.name == "periodo"
    assert list(df.index) == [PeriodoMensual(2026, 2), PeriodoMensual(2026, 3)]  # ordenado
    # float("145.200") directo del parseo en _fetch, sin cálculo de por medio —
    # igualdad exacta, no tolerancia (145.2001 pasaría el default de approx).
    assert df["INPP sin Petróleo y con Servicios"][PeriodoMensual(2026, 3)] == 145.200  # type: ignore[call-overload]
    # todas las columnas del tipo — no un subconjunto
    assert "Demanda interna" in df.columns
    assert "81 Otros servicios excepto actividades gubernamentales" in df.columns


def test_consultar_indice_tipo_invalido_lanza_error_configuracion() -> None:
    with pytest.raises(ErrorConfiguracion):
        consultas.consultar_indice("tipo_inexistente")
