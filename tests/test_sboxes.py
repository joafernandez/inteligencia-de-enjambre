"""Integridad de los datos del artículo y reproducción de sus métricas."""
import pytest

from src.metricas import calcular_propiedades
from src.sboxes import cargar_metricas_reportadas, cargar_sboxes, fuente

SBOXES = {s.id: s for s in cargar_sboxes()}
REP = cargar_metricas_reportadas()


def test_siete_sboxes_biyectivas():
    assert list(SBOXES) == ["CSO", "PSO", "ANO", "TLBO", "ABC", "GA", "FA"]
    for s in SBOXES.values():
        assert sorted(s.valores) == list(range(256))


def test_valores_coinciden_con_hex_publicado():
    for s in SBOXES.values():
        assert [int(h, 16) for h in s.hex_publicado] == list(s.valores)


def test_primera_fila_tabla1_articulo():
    assert SBOXES["CSO"].hex_publicado[:4] == ("8", "48", "72", "9C")


def test_fuente_registrada():
    assert fuente()["doi"] == "10.1038/s41598-025-25173-3"


@pytest.mark.parametrize("sid", list(SBOXES))
def test_uniformidad_diferencial_reproduce_articulo(sid):
    assert calcular_propiedades(SBOXES[sid].valores).du == REP.loc[sid, "du"]


@pytest.mark.parametrize("sid", list(SBOXES))
def test_bic_nl_minimo_reproduce_articulo(sid):
    assert calcular_propiedades(SBOXES[sid].valores).bic_nl_min == REP.loc[sid, "bic_nl_min"]


@pytest.mark.parametrize("sid", ["CSO", "PSO", "ANO", "ABC", "GA"])
def test_nonlinealidad_reproduce_articulo(sid):
    p = calcular_propiedades(SBOXES[sid].valores)
    assert (p.nl_min, p.nl_max, p.nl_prom) == tuple(REP.loc[sid, ["nl_min", "nl_max", "nl_prom"]])


@pytest.mark.parametrize("sid", ["CSO", "PSO", "ANO", "GA", "FA"])
def test_lp_reproduce_articulo(sid):
    assert calcular_propiedades(SBOXES[sid].valores).lp == pytest.approx(REP.loc[sid, "lp"], abs=1e-4)


@pytest.mark.parametrize("sid", ["CSO", "PSO", "ANO", "ABC", "GA"])
def test_promedios_sac_y_bic_sac_reproducen_articulo(sid):
    p = calcular_propiedades(SBOXES[sid].valores)
    assert p.sac_prom == pytest.approx(REP.loc[sid, "sac_prom"], abs=2e-4)
    assert p.bic_sac_prom == pytest.approx(REP.loc[sid, "bic_sac_prom"], abs=2e-4)


def test_discrepancias_documentadas_se_mantienen():
    """Estas diferencias con el artículo están documentadas en docs/; si cambian,
    hay que revisar la documentación."""
    tlbo = calcular_propiedades(SBOXES["TLBO"].valores)
    assert (tlbo.nl_min, tlbo.nl_max) == (100, 106)        # artículo: 104 / 110
    assert calcular_propiedades(SBOXES["ABC"].valores).lp == 32 / 256  # artículo: 0.1171875
