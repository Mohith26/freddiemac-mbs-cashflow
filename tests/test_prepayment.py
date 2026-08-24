import json
from pathlib import Path

import numpy as np
import pytest

from poolflow.prepayment import cpr_to_smm, psa_cpr, psa_smm, smm_to_cpr

FIXTURES = json.loads(
    (Path(__file__).resolve().parent.parent / "fixtures" / "psa_fixtures.json").read_text()
)["fixtures"]


class TestConversions:
    def test_cpr_to_smm_formula_exact(self):
        cpr = 0.06
        assert cpr_to_smm(cpr) == pytest.approx(1 - (1 - 0.06) ** (1 / 12), abs=0)

    def test_round_trip_cpr(self):
        for cpr in [0.0, 0.001, 0.06, 0.25, 0.90, 1.0]:
            assert smm_to_cpr(cpr_to_smm(cpr)) == pytest.approx(cpr, abs=1e-14)

    def test_round_trip_smm(self):
        for smm in [0.0, 0.0005, 0.005143012832, 0.02, 0.5]:
            assert cpr_to_smm(smm_to_cpr(smm)) == pytest.approx(smm, abs=1e-14)

    def test_zero_maps_to_zero(self):
        assert cpr_to_smm(0.0) == 0.0
        assert smm_to_cpr(0.0) == 0.0

    def test_one_maps_to_one(self):
        assert cpr_to_smm(1.0) == 1.0
        assert smm_to_cpr(1.0) == 1.0

    def test_vectorized_conversion(self):
        cprs = np.array([0.01, 0.06, 0.12])
        smms = cpr_to_smm(cprs)
        assert smms.shape == (3,)
        np.testing.assert_allclose(smm_to_cpr(smms), cprs, atol=1e-14)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_raises(self, bad):
        with pytest.raises(ValueError):
            cpr_to_smm(bad)
        with pytest.raises(ValueError):
            smm_to_cpr(bad)


class TestPSA:
    @pytest.mark.parametrize(
        "fx", FIXTURES, ids=[f"psa{f['psa']}_age{f['age']}" for f in FIXTURES]
    )
    def test_hand_computed_fixture_cpr(self, fx):
        assert psa_cpr(fx["age"], fx["psa"]) == pytest.approx(fx["cpr"], abs=1e-12)

    @pytest.mark.parametrize(
        "fx", FIXTURES, ids=[f"psa{f['psa']}_age{f['age']}" for f in FIXTURES]
    )
    def test_hand_computed_fixture_smm(self, fx):
        assert psa_smm(fx["age"], fx["psa"]) == pytest.approx(fx["smm"], abs=5e-13)

    def test_ramp_is_linear_through_month_30(self):
        for age in range(1, 31):
            assert psa_cpr(age, 100) == pytest.approx(0.002 * age, abs=1e-15)

    def test_plateau_after_month_30(self):
        assert psa_cpr(31, 100) == psa_cpr(30, 100) == psa_cpr(360, 100)

    def test_multiplier_scales_cpr_linearly(self):
        assert psa_cpr(10, 200) == pytest.approx(2 * psa_cpr(10, 100), abs=1e-15)
        assert psa_cpr(10, 50) == pytest.approx(0.5 * psa_cpr(10, 100), abs=1e-15)

    def test_zero_psa_means_no_prepayment(self):
        ages = np.arange(1, 361)
        assert np.all(psa_cpr(ages, 0.0) == 0.0)

    def test_extreme_multiplier_clips_at_full_prepayment(self):
        assert psa_cpr(30, 100_000) == 1.0

    def test_vectorized_ages(self):
        ages = np.array([1, 15, 30, 100])
        out = psa_cpr(ages, 100)
        np.testing.assert_allclose(out, [0.002, 0.030, 0.060, 0.060], atol=1e-15)

    def test_age_zero_raises(self):
        with pytest.raises(ValueError):
            psa_cpr(0, 100)

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError):
            psa_cpr(10, -50)
