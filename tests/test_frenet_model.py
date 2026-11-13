
from pathlib import Path

import casadi as ca
import numpy as np

from latch13.models.frenet_model import build_frenet_model
from Prediction_Models.pred_model_dynamic_stm_pacejka import pred_stm


ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "external/TUM-CONTROL/Config/EDGAR"
VEHICLE = CONFIG / "veh_params_pred.yaml"
TIRES = CONFIG / "pacejka_params.yaml"


def test_frenet_dynamics():
    frenet = build_frenet_model(VEHICLE, TIRES)
    tum, _ = pred_stm(str(VEHICLE), str(TIRES))

    f_frenet = ca.Function(
        "f_frenet_test",
        [frenet.x, frenet.u, frenet.p],
        [frenet.f_expl_expr],
    )

    f_tum = ca.Function(
        "f_tum_test",
        [tum.x, tum.u],
        [tum.xdot],
    )

    # Test multiple physically reasonable operating points
    for vx in [5.0, 10.0, 15.0]:
        for delta in [-0.05, 0.0, 0.05]:

            xf = np.array([
                0.0,    # s
                0.1,    # lateral error
                0.03,   # heading error
                vx,
                0.1,    # lateral velocity
                0.02,   # yaw rate
                delta,
                0.2,    # acceleration
            ])

            u = np.array([0.01, 0.1])
            kappa = np.array([0.001])

            xc = np.array([
                0.0, 0.0, 0.0,
                *xf[3:],
            ])

            df = np.asarray(
                f_frenet(xf, u, kappa)
            ).flatten()

            dc = np.asarray(
                f_tum(xc, u)
            ).flatten()

            # Physical dynamics must be identical
            np.testing.assert_allclose(
                df[3:],
                dc[3:],
                rtol=1e-9,
                atol=1e-9,
            )

            assert np.all(np.isfinite(df))


def test_model_dimensions():
    model = build_frenet_model(VEHICLE, TIRES)

    assert model.x.numel() == 8
    assert model.u.numel() == 2
    assert model.p.numel() == 1
    assert model.f_expl_expr.numel() == 8
