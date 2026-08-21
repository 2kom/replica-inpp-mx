from __future__ import annotations

import os
import traceback
from unittest.mock import MagicMock

import pytest
import requests

from replica_inpp.api import consultas
from replica_inpp.dominio.errores import (
    ErrorConfiguracion,
    FuenteNoDisponible,
    InvarianteViolado,
    RespuestaInvalida,
)
from replica_inpp.dominio.periodos import PeriodoMensual
from replica_inpp.infraestructura.inegi.fuente_validacion_api import (
    _INDICADORES,
    FuenteValidacionApi,
)

_TIPO = "PRODUCCION TOTAL"
# Cantidad real de indicadores del tipo — no hardcodear un número aparte, evita que
# el test quede desincronizado si el dict crece (ver docs/requerimientos/indicadores_bie_inpp.md).
_N_INDICADORES = len(_INDICADORES[_TIPO])

_PM1 = PeriodoMensual(2026, 3)
_PM2 = PeriodoMensual(2026, 2)

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

_RESPUESTA_MENSUAL_CON_NULL = {
    "Series": [
        {
            "OBSERVATIONS": [
                {"TIME_PERIOD": "2026/03", "OBS_VALUE": None, "OBS_STATUS": "3"},
                {"TIME_PERIOD": "2026/02", "OBS_VALUE": "144.300", "OBS_STATUS": "3"},
            ]
        }
    ]
}


@pytest.fixture(autouse=True)
def limpiar_cache():
    FuenteValidacionApi._cache.clear()
    yield
    FuenteValidacionApi._cache.clear()


def _mock_resp(status_code: int, json_data) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


class TestInicializacion:
    def test_tipo_invalido_lanza_error_configuracion(self):
        with pytest.raises(ErrorConfiguracion):
            FuenteValidacionApi(token="cualquier-token", tipo="tipo_inexistente")

    def test_tipo_valido_no_lanza(self):
        FuenteValidacionApi(token="cualquier-token", tipo=_TIPO)

    @pytest.mark.parametrize(
        "timeout",
        [0, -1, -10, float("nan"), float("inf"), float("-inf")],
        ids=["cero", "negativo", "negativo_grande", "nan", "inf", "menos_inf"],
    )
    def test_timeout_no_positivo_lanza_error_configuracion(self, timeout):
        # timeout<=0 no atrapa NaN ni inf (ambas comparaciones dan False) — la
        # guardia real exige valor finito Y positivo, no solo "no <= 0".
        with pytest.raises(ErrorConfiguracion, match="timeout"):
            FuenteValidacionApi(token="cualquier-token", tipo=_TIPO, timeout=timeout)

    def test_timeout_positivo_no_lanza(self):
        FuenteValidacionApi(token="cualquier-token", tipo=_TIPO, timeout=1)


class TestObtenerIndices:
    def test_devuelve_valores_para_periodos_pedidos(self, mocker):
        mocker.patch("requests.get", return_value=_mock_resp(200, _RESPUESTA_MENSUAL))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        resultado = fuente.obtener_indices([_PM1, _PM2])

        assert resultado["INPP sin Petróleo y con Servicios"][_PM1] == pytest.approx(145.200)
        assert resultado["INPP sin Petróleo y con Servicios"][_PM2] == pytest.approx(144.300)
        assert resultado["Demanda interna"][_PM1] == pytest.approx(145.200)

    @pytest.mark.parametrize(
        "fuera_de_rango",
        [PeriodoMensual(2000, 1), PeriodoMensual(2030, 1)],
        ids=["anterior_al_historico", "posterior_al_historico"],
    )
    def test_periodo_fuera_del_historico_se_omite(self, mocker, fuera_de_rango):
        mocker.patch("requests.get", return_value=_mock_resp(200, _RESPUESTA_MENSUAL))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        resultado = fuente.obtener_indices([fuera_de_rango, _PM1])

        assert fuera_de_rango not in resultado["INPP sin Petróleo y con Servicios"]
        assert resultado["INPP sin Petróleo y con Servicios"][_PM1] == pytest.approx(145.200)

    def test_obs_value_null_devuelve_none(self, mocker):
        mocker.patch("requests.get", return_value=_mock_resp(200, _RESPUESTA_MENSUAL_CON_NULL))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        resultado = fuente.obtener_indices([_PM1, _PM2])

        assert resultado["INPP sin Petróleo y con Servicios"][_PM1] is None
        assert resultado["INPP sin Petróleo y con Servicios"][_PM2] == pytest.approx(144.300)

    def test_usa_los_ids_reales_de_cada_nombre(self, mocker):
        mock_get = mocker.patch("requests.get", return_value=_mock_resp(200, _RESPUESTA_MENSUAL))
        FuenteValidacionApi(token="token", tipo=_TIPO).obtener_indices([_PM1])

        urls = [llamada.args[0] for llamada in mock_get.call_args_list]
        assert any("910491" in url for url in urls)  # INPP sin Petróleo y con Servicios
        assert any("1380015" in url for url in urls)  # Demanda interna
        assert mock_get.call_count == _N_INDICADORES

    def test_timeout_del_constructor_se_pasa_a_requests_get(self, mocker):
        mock_get = mocker.patch("requests.get", return_value=_mock_resp(200, _RESPUESTA_MENSUAL))
        FuenteValidacionApi(token="token", tipo=_TIPO, timeout=42).obtener_indices([_PM1])
        assert mock_get.call_args.kwargs["timeout"] == 42

    def test_timeout_default_es_10(self, mocker):
        mock_get = mocker.patch("requests.get", return_value=_mock_resp(200, _RESPUESTA_MENSUAL))
        FuenteValidacionApi(token="token", tipo=_TIPO).obtener_indices([_PM1])
        assert mock_get.call_args.kwargs["timeout"] == 10


class TestCache:
    def test_segunda_llamada_no_repite_ningun_request(self, mocker):
        mock_get = mocker.patch("requests.get", return_value=_mock_resp(200, _RESPUESTA_MENSUAL))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        fuente.obtener_indices([_PM1])
        fuente.obtener_indices([_PM2])

        assert mock_get.call_count == _N_INDICADORES

    def test_cache_compartido_entre_instancias(self, mocker):
        mock_get = mocker.patch("requests.get", return_value=_mock_resp(200, _RESPUESTA_MENSUAL))

        FuenteValidacionApi(token="token", tipo=_TIPO).obtener_indices([_PM1])
        FuenteValidacionApi(token="token", tipo=_TIPO).obtener_indices([_PM1])

        assert mock_get.call_count == _N_INDICADORES


class TestApiNoDisponible:
    def test_timeout_lanza_fuente_no_disponible(self, mocker):
        mocker.patch("requests.get", side_effect=requests.exceptions.Timeout("timeout"))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(FuenteNoDisponible):
            fuente.obtener_indices([_PM1])

    def test_http_400_lanza_fuente_no_disponible(self, mocker):
        mock_resp = _mock_resp(400, {})
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("400")
        mocker.patch("requests.get", return_value=mock_resp)

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(FuenteNoDisponible):
            fuente.obtener_indices([_PM1])

    def test_error_no_expone_token_en_mensaje_ni_traceback(self, mocker):
        # La URL real de la API lleva el token en texto plano; un HTTPError de
        # requests incluye la URL completa en su propio mensaje. `_fetch` no debe
        # dejarlo pasar ni en el mensaje de FuenteNoDisponible ni encadenado como
        # causa (eso también lo imprime el traceback por defecto).
        resp = requests.Response()
        resp.status_code = 401
        resp.reason = "Unauthorized"
        resp.url = (
            "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/"
            "INDICATOR/910491/es/00/false/BIE-BISE/2.0/TOKEN_SECRETO?type=json"
        )
        mocker.patch("requests.get", return_value=resp)

        fuente = FuenteValidacionApi(token="TOKEN_SECRETO", tipo=_TIPO)
        with pytest.raises(FuenteNoDisponible) as exc_info:
            fuente.obtener_indices([_PM1])

        assert "TOKEN_SECRETO" not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        traza = "".join(
            traceback.format_exception(
                type(exc_info.value), exc_info.value, exc_info.value.__traceback__
            )
        )
        assert "TOKEN_SECRETO" not in traza

    def test_connection_error_no_expone_token_en_mensaje_ni_traceback(self, mocker):
        mocker.patch(
            "requests.get",
            side_effect=requests.exceptions.ConnectionError(
                "No se pudo conectar a https://www.inegi.org.mx/.../2.0/TOKEN_SECRETO?type=json"
            ),
        )

        fuente = FuenteValidacionApi(token="TOKEN_SECRETO", tipo=_TIPO)
        with pytest.raises(FuenteNoDisponible) as exc_info:
            fuente.obtener_indices([_PM1])

        assert "TOKEN_SECRETO" not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None


class TestRespuestaInvalida:
    def test_json_invalido_lanza_respuesta_invalida(self, mocker):
        mock_resp = _mock_resp(200, {})
        mock_resp.json.side_effect = ValueError("no es json")
        mocker.patch("requests.get", return_value=mock_resp)

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    def test_json_raiz_no_es_objeto_lanza_respuesta_invalida(self, mocker):
        # JSON raíz una lista en vez de un objeto — antes producía TypeError
        # nativo al indexar data["Series"].
        mocker.patch("requests.get", return_value=_mock_resp(200, []))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    def test_sin_clave_series_lanza_respuesta_invalida(self, mocker):
        mocker.patch("requests.get", return_value=_mock_resp(200, {"Header": {}}))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    def test_series_vacio_lanza_respuesta_invalida(self, mocker):
        mocker.patch("requests.get", return_value=_mock_resp(200, {"Series": []}))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    def test_series_no_es_lista_lanza_respuesta_invalida(self, mocker):
        mocker.patch("requests.get", return_value=_mock_resp(200, {"Series": "no-es-lista"}))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    def test_observations_vacio_lanza_respuesta_invalida(self, mocker):
        mocker.patch(
            "requests.get", return_value=_mock_resp(200, {"Series": [{"OBSERVATIONS": []}]})
        )

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida, match="OBSERVATIONS"):
            fuente.obtener_indices([_PM1])

    def test_observations_no_es_lista_lanza_respuesta_invalida(self, mocker):
        mocker.patch(
            "requests.get",
            return_value=_mock_resp(200, {"Series": [{"OBSERVATIONS": "no-es-lista"}]}),
        )

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    def test_observacion_no_es_objeto_lanza_respuesta_invalida(self, mocker):
        respuesta = {"Series": [{"OBSERVATIONS": ["no-es-un-objeto"]}]}
        mocker.patch("requests.get", return_value=_mock_resp(200, respuesta))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    def test_time_period_no_es_texto_lanza_respuesta_invalida(self, mocker):
        # TIME_PERIOD numérico — antes producía AttributeError nativo al llamar
        # .split() sobre un int.
        respuesta = {"Series": [{"OBSERVATIONS": [{"TIME_PERIOD": 202501, "OBS_VALUE": "100"}]}]}
        mocker.patch("requests.get", return_value=_mock_resp(200, respuesta))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida, match="TIME_PERIOD"):
            fuente.obtener_indices([_PM1])

    def test_time_period_malformado_lanza_respuesta_invalida(self, mocker):
        respuesta = {
            "Series": [{"OBSERVATIONS": [{"TIME_PERIOD": "formato-malo", "OBS_VALUE": "145.0"}]}]
        }
        mocker.patch("requests.get", return_value=_mock_resp(200, respuesta))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    def test_time_period_quincenal_lanza_respuesta_invalida(self, mocker):
        # El INPP es solo mensual — un TIME_PERIOD de 3 partes (quincenal, como
        # en el INPC) debe rechazarse, no aceptarse silenciosamente.
        respuesta = {
            "Series": [{"OBSERVATIONS": [{"TIME_PERIOD": "2026/03/01", "OBS_VALUE": "145.0"}]}]
        }
        mocker.patch("requests.get", return_value=_mock_resp(200, respuesta))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    def test_time_period_año_mes_no_numericos_lanza_respuesta_invalida(self, mocker):
        respuesta = {
            "Series": [{"OBSERVATIONS": [{"TIME_PERIOD": "AAAA/BB", "OBS_VALUE": "145.0"}]}]
        }
        mocker.patch("requests.get", return_value=_mock_resp(200, respuesta))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    def test_obs_value_ausente_lanza_respuesta_invalida(self, mocker):
        respuesta = {"Series": [{"OBSERVATIONS": [{"TIME_PERIOD": "2026/03"}]}]}
        mocker.patch("requests.get", return_value=_mock_resp(200, respuesta))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida, match="OBS_VALUE"):
            fuente.obtener_indices([_PM1])

    def test_obs_value_no_numerico_lanza_respuesta_invalida(self, mocker):
        respuesta = {
            "Series": [{"OBSERVATIONS": [{"TIME_PERIOD": "2026/03", "OBS_VALUE": "no-es-numero"}]}]
        }
        mocker.patch("requests.get", return_value=_mock_resp(200, respuesta))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida):
            fuente.obtener_indices([_PM1])

    @pytest.mark.parametrize("valor", ["NaN", "1e999", "-1e999"], ids=["nan", "inf", "menos_inf"])
    def test_obs_value_no_finito_lanza_respuesta_invalida(self, mocker, valor):
        respuesta = {"Series": [{"OBSERVATIONS": [{"TIME_PERIOD": "2026/03", "OBS_VALUE": valor}]}]}
        mocker.patch("requests.get", return_value=_mock_resp(200, respuesta))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        with pytest.raises(RespuestaInvalida, match="finito"):
            fuente.obtener_indices([_PM1])


class TestHistoricoIndices:
    def test_incluye_huecos_internos_como_none(self, mocker):
        # Ene y Mar publicados, Feb ausente de la respuesta — debe aparecer con
        # None en vez de faltar (a diferencia de un periodo fuera de rango).
        respuesta = {
            "Series": [
                {
                    "OBSERVATIONS": [
                        {"TIME_PERIOD": "2026/01", "OBS_VALUE": "140.0"},
                        {"TIME_PERIOD": "2026/03", "OBS_VALUE": "145.2"},
                    ]
                }
            ]
        }
        mocker.patch("requests.get", return_value=_mock_resp(200, respuesta))

        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)
        resultado = fuente.historico_indices()["INPP sin Petróleo y con Servicios"]

        assert list(resultado) == [
            PeriodoMensual(2026, 1),
            PeriodoMensual(2026, 2),
            PeriodoMensual(2026, 3),
        ]
        assert resultado[PeriodoMensual(2026, 2)] is None


class TestPeriodosVacios:
    # El puerto promete InvarianteViolado con `periodos == []`. La guardia va
    # antes de tocar cache o red, así que call_count debe quedar en 0.
    def test_lista_vacia_lanza_invariante_sin_hacer_request(self, mocker):
        mock_get = mocker.patch("requests.get")
        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)

        with pytest.raises(InvarianteViolado, match="obtener_indices"):
            fuente.obtener_indices([])

        assert mock_get.call_count == 0

    def test_mensaje_nombra_el_parametro_vacio(self, mocker):
        mocker.patch("requests.get")
        fuente = FuenteValidacionApi(token="token", tipo=_TIPO)

        with pytest.raises(InvarianteViolado) as exc:
            fuente.obtener_indices([])

        assert "periodos" in str(exc.value)
        assert "vacío" in str(exc.value)


class TestSmokeApiReal:
    """Smoke test contra el BIE real — no se corre en CI ni por defecto.

    Requiere `INEGI_TOKEN` en el entorno; se salta si no está configurado.
    """

    @pytest.mark.requires_api
    def test_consultar_indice_contra_api_real(self, monkeypatch: pytest.MonkeyPatch):
        token = os.environ.get("INEGI_TOKEN")
        if not token:
            pytest.skip("INEGI_TOKEN no configurado")
        monkeypatch.setenv("INEGI_TOKEN", token)

        df = consultas.consultar_indice("produccion total")

        assert not df.empty
        assert "INPP sin Petróleo y con Servicios" in df.columns
