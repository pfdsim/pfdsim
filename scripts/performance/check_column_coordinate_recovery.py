#!/usr/bin/env python3
"""Check runtime patch restoration and projection invariants against saved roots.

No nonlinear benchmark solves are restarted. Example from repository root:
  OPENBLAS_NUM_THREADS=1 .venv/bin/python \
      scripts/performance/check_column_coordinate_recovery.py \
      --records /tmp/pfdsim-coordinate-portfolio-20261009
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))

from diagnose_binary_azeotropic_overdraw import capture_problem
from probe_azeotropic_overdistillation import prepare_case
from probe_column_coordinate_recovery import (
    InventoryCoordinates,RuntimePatch,StableLogCoordinates,binary_region,material_update,
)


def main():
    import numpy as np
    from scipy.sparse import csr_matrix
    from equilibrium_stage_column import EquilibriumStageColumnMixin
    from unit_operations_distillation import RigorousDistillation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records",type=Path,required=True)
    args = parser.parse_args()
    original_solver = EquilibriumStageColumnMixin._sparse_newton_solve
    original_model = EquilibriumStageColumnMixin._build_mesh_model
    unit = EquilibriumStageColumnMixin()
    unit.unit_id = "dispatch_probe"
    unit.thermo = SimpleNamespace()
    options = dict(mesh_tolerance=1e-6,max_iterations=4,max_jacobian_evaluations=4,
                   line_search_steps=16,finite_difference_rel_step=1e-6)
    def residual(z):
        return np.array([np.exp(z[0])-1e-5,z[1]-2.])
    def jacobian(z,f,h):
        return csr_matrix([[np.exp(z[0]),0.],[0.,1.]]),0,"analytic"
    attempts = []
    with RuntimePatch(4,attempts,"coordinate_cap"):
        answer = unit._sparse_newton_solve(residual,csr_matrix(np.eye(2)),np.array([-60.,0.]),options,jacobian=jacobian)
        assert attempts[0]["algorithm"]=="coordinate_cap"
        assert abs(answer["x"][1]-2.)<1e-10
    assert EquilibriumStageColumnMixin._sparse_newton_solve is original_solver
    assert EquilibriumStageColumnMixin._build_mesh_model is original_model
    print("PASS coordinate-cap dispatch and useful-step retention; runtime methods restored")

    rng = np.random.default_rng(20261009)
    checked = 0
    for case in ("ethanol_water","ipa_water"):
        records = [json.loads(line) for line in (args.records/(case+"-results.jsonl")).read_text().splitlines()]
        base,inlets = prepare_case(case)
        for record in records:
            if not record["success"]:
                continue
            unit = RigorousDistillation(base.unit_id,base.thermo,record["params"])
            p = capture_problem(unit,inlets)
            z = np.array(record["attempts"][-1]["native_vector"])
            raw = p["residual"](z)
            assert np.linalg.norm(raw,ord=np.inf)<record["params"]["mesh_tolerance"]
            stable = StableLogCoordinates(unit,p["model"],p["model_args"],z)
            binary_region(stable)
            u = stable.pack(z)
            zz = stable.native(u)
            ff = p["residual"](zz)
            assert np.linalg.norm(ff,ord=np.inf)<2e-6
            x0 = p["model"]["decode"](z)["x"][-1][stable.comps[0]]
            x1 = p["model"]["decode"](zz)["x"][-1][stable.comps[0]]
            assert np.isclose(x0,x1,rtol=1e-8,atol=1e-300)
            entry = {}
            material_update(stable,u,entry,zz)
            assert entry["max_linear_inventory_error"]<1e-9
            scaled_energy = stable.P@ff
            interior = [j*(stable.nc+2)+stable.nc for j in range(1,stable.N-1)]
            assert np.max(np.abs(scaled_energy[interior]))<1e-9
            projected = InventoryCoordinates(unit,p["model"],p["model_args"],z)
            start = projected.pack(z)
            for _ in range(10):
                trial = start.copy()
                trial[projected.N:projected.new_flow_start] = rng.uniform(.01,.99,projected.new_flow_start-projected.N)
                trial[projected.new_flow_start:projected.new_Q_start] = rng.uniform(.1,2.,projected.N-2)
                native = projected.native(trial)
                f = p["residual"](native)
                a = p["model_args"]
                totals = [sum(f[j*(projected.nc+2)+ci]*a["component_scales"][c] for ci,c in enumerate(projected.comps))
                          for j in range(projected.N)]
                assert max(abs(v) for v in totals)<1e-9
                assert np.max(np.abs(f[projected.N*(projected.nc+2):]))<1e-10
                d = p["model"]["decode"](native)
                for ci,c in enumerate(projected.comps):
                    amount = d["V"][0]*d["x"][0][c]+d["L"][-1]*d["x"][-1][c]
                    assert abs(amount-projected.inventory[ci])<1e-10
            checked += 1
            print("PASS",case,record["factor"],"native root, stable trace-inventory round trip, energy elimination, material solve, 10 random projections")
    assert checked>=2
    print("PASS",checked,"saved solutions; no nonlinear benchmark reruns")


if __name__=="__main__":
    main()
