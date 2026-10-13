"""Pair-weighted raw VLE comparison of five UNIFAC parameter families.

Uses measured P/T/x/y collected by collect_raw_vle_benchmark.py. Predictions
reuse PFDSim's activity, gamma-phi, and source-property paths at measured states.
No bubble-pressure or temperature root solves are performed.
No liquid interaction fitting occurs. Each binary pair has equal weight;
points within a pair have equal weight regardless of paper/dataset size.
Missing structural/interaction or vapor/property coverage is explicitly saved.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from types import MethodType
import time

import numpy as np
from chemicals.acentric import omega as reference_omega
from chemicals.critical import Tc, Pc, Vc
from chemicals.dipole import dipole_moment
from chemicals.vapor_pressure import (
    Antoine, Wagner, Psat_data_WagnerPoling, Psat_data_Landolt_Antoine,
    Psat_data_VDI_PPDS_3,
)
from chemicals.dippr import EQ101
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.activity_fitting.benchmark_unifac_runtime_fits import IdentityDatabase, METHODS, write_csv
from perry_properties import PerryPropertyLibrary
from nannoolal_method import estimate, acentric_factor
from thermodynamics_models.gamma_phi import UNIFACBVThermodynamics
from thermodynamics_models.second_virial import create_second_virial_provider

SOURCE = ROOT / "data/source/activity_fitting/raw_vle_measurements.json"


class SourceCurves:
    """Shared, source-backed pure Psat; selection never depends on UNIFAC."""

    def __init__(self, source):
        self.source = source
        self.perry = PerryPropertyLibrary()
        self.used = {}

    def evaluate(self, doi, cas, t):
        spec = self.source["source_psat"].get(doi, {}).get(cas)
        if spec is not None:
            if spec.get("type") == "dippr101":
                if not spec["Tmin_K"] <= t <= spec["Tmax_K"]:
                    raise ValueError("source DIPPR temperature range exceeded")
                pressure = EQ101(t, *spec["coefficients"]) * spec["factor_Pa"]
            else:
                pressure = Antoine(t, spec["A"], spec["B"], spec["C_K"], base=spec["base"]) * spec["factor_Pa"]
            label = "source paper correlation"
        elif doi == "10.3390/pr13072034":
            from scripts.activity_fitting.fit_fendu_2025_activity_parameters import psat_kpa
            name = {"78-83-1":"isobutanol", "71-36-3":"1-butanol", "104-76-7":"2-ethyl-1-hexanol"}[cas]
            pressure = float(psat_kpa(name, t))*1000
            label = "source pure-measurement Antoine fit (existing Fendu helper)"
        else:
            perry = self.perry.vapor_pressure_bar(cas, t)
            if perry is not None:
                pressure = perry.value*100000
                label = "Perry 2-8 in-range"
            elif cas in Psat_data_WagnerPoling.index:
                p = Psat_data_WagnerPoling.loc[cas]
                if not p["Tmin"] <= t <= min(p["Tmax"], p["Tc"]):
                    raise ValueError("Wagner source temperature range exceeded")
                pressure = Wagner(t, p["Tc"], p["Pc"], p["A"], p["B"], p["C"], p["D"])
                label = "Poling Wagner in-range"
            elif cas == "56-81-5" or cas == "111-43-3":
                # PPDS temperature domain begins at its source melting point.
                p = Psat_data_VDI_PPDS_3.loc[cas]
                if not p["Tm"] <= t < p["Tc"]:
                    raise ValueError("VDI PPDS liquid-vapor domain exceeded")
                pressure = Wagner(t, p["Tc"], p["Pc"], p["A"], p["B"], p["C"], p["D"])
                label = "VDI PPDS liquid-vapor curve"
            elif cas in Psat_data_Landolt_Antoine.index:
                p = Psat_data_Landolt_Antoine.loc[cas]
                if not p["Tmin"] <= t <= p["Tmax"]:
                    raise ValueError("Landolt Antoine source temperature range exceeded")
                pressure = Antoine(t, p["A"], p["B"], p["C"], base=math.e)
                label = "Landolt Antoine in-range"
            else:
                raise ValueError(f"no qualified pure vapor pressure for {cas} at {t:g} K")
        if not math.isfinite(pressure) or pressure <= 0:
            raise ValueError("source vapor pressure is non-positive/non-finite")
        self.used[doi,cas] = label
        return pressure/100000


class PropertyDatabase(IdentityDatabase):
    """Identity plus shared vapor-model inputs, without online hydration."""

    def __init__(self):
        self.local = {r["CAS"]:r for r in json.loads((ROOT / "data/chemicals.json").read_text())["chemicals"].values() if r.get("CAS")}
        self.perry = PerryPropertyLibrary()

    @lru_cache(maxsize=None)
    def get(self, cas):
        p = deepcopy(super().get(cas))
        row = self.local.get(cas, {})
        critical = (self.perry.get(cas) or {}).get("critical_constants", {})
        p.Tc = critical.get("Tc_K") or row.get("Tc") or Tc(cas)
        pc = critical.get("Pc_MPa")
        p.Pc = pc*10 if pc else row.get("Pc") or ((Pc(cas) or 0)/100000 or None)
        vc = critical.get("Vc_m3_per_kmol")
        p.Vc = vc*1000 if vc else row.get("Vc") or ((Vc(cas) or 0)*1e6 or None)
        p.omega = critical.get("omega", row.get("omega", reference_omega(cas)))
        p.Zc = critical.get("Zc") or row.get("Zc")
        p.dipole_moment = row.get("dipole_moment") or dipole_moment(cas)
        if None in (p.Tc,p.Pc,p.Vc,p.omega):
            prediction = estimate(p.smiles)
            p.Tc = p.Tc or prediction.tc_K
            p.Pc = p.Pc or (prediction.pc_kPa/100 if prediction.pc_kPa else None)
            p.Vc = p.Vc or prediction.vc_cm3_mol
            if p.omega is None:
                p.omega = acentric_factor(p.smiles,tc=p.Tc,pc_kPa=p.Pc*100 if p.Pc else None)
            p.property_sources["benchmark_critical_fallback"] = {"source":"Nannoolal estimate","note":"used only for common vapor correction"}
        return p


def is_acid(db, pair):
    pattern = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    return any(Chem.MolFromSmiles(db.get(c).smiles).HasSubstructMatch(pattern) for c in pair)


def pair_metrics(rows):
    result = {"points":len(rows), "successes":sum(r["status"] == "ok" for r in rows)}
    good = [r for r in rows if r["status"] == "ok"]
    if not good:
        return result
    closure=[r for r in good if 'bubble_equation_ln_residual' in r]
    if closure:
        errors = np.array([r['bubble_equation_ln_residual'] for r in closure])
        result.update(bubble_equation_points=len(closure),
                      bubble_equation_AARD_percent=float(np.mean([abs(r['bubble_equation_relative_residual']) for r in closure]))*100,
                      bubble_equation_ln_MAE=float(np.mean(np.abs(errors))), bubble_equation_ln_MSE=float(np.mean(errors**2)))
    lng = [r[k] for r in good for k in ("ln_gamma_error1","ln_gamma_error2") if k in r]
    if lng:
        result.update(ln_gamma_values=len(lng), ln_gamma_MAE=float(np.mean(np.abs(lng))), ln_gamma_MSE=float(np.mean(np.square(lng))))
    return result


def measured_state(model,measurement,reference_cache):
    """Evaluate fugacity relations at the measured state; no P/T root solves."""
    pair=model.components
    t,p,x1=measurement['T_K'],measurement['P_Pa']/100000,measurement['x1']
    x={pair[0]:x1,pair[1]:1-x1}
    gamma=model.activity_coefficients(t,x)
    cache_key=(measurement['doi'],t,p)
    if cache_key not in reference_cache:
        reference_cache[cache_key]=model._gamma_phi_reference_factors(t,p)
    reference=reference_cache[cache_key]
    target={c:x[c]*gamma[c]*reference[c]/p for c in pair}
    result={f'predicted_ln_gamma{i}':math.log(gamma[c]) for i,c in enumerate(pair,1)}
    if 0<measurement.get('y1',0)<1:
        y={pair[0]:measurement['y1'],pair[1]:1-measurement['y1']}
        phi=model.vapor_fugacity_coefficients(t,p,y)
        for i,c in enumerate(pair,1):
            inferred=math.log(y[c]*phi[c]*p/(x[c]*reference[c]))
            result[f'inferred_ln_gamma{i}']=inferred
            result[f'ln_gamma_error{i}']=math.log(gamma[c])-inferred
    else:
        total=sum(target.values())
        y={c:target[c]/total for c in pair}
        for _ in range(80):
            phi=model.vapor_fugacity_coefficients(t,p,y)
            amounts={c:target[c]/phi[c] for c in pair}
            total=sum(amounts.values())
            new_y={c:amounts[c]/total for c in pair}
            if max(abs(new_y[c]-y[c]) for c in pair)<1e-10:
                y=new_y
                break
            y={c:.5*(y[c]+new_y[c]) for c in pair}
        else:raise ValueError('fixed-P vapor composition did not converge')
        phi=model.vapor_fugacity_coefficients(t,p,y)
    closure=sum(target[c]/phi[c] for c in pair)
    result.update(bubble_equation_relative_residual=closure-1,bubble_equation_ln_residual=math.log(closure))
    return result


def evaluate_pair(pair, rows, source, output):
    """Independent worker writes all method results for this one CAS pair."""
    db, curves = PropertyDatabase(), SourceCurves(source)
    audit, results, model_info, reference_cache = [], [], [], {}
    try:
        acid = is_acid(db, pair)
        properties = {c:db.get(c) for c in pair}
        correlation = "HOC" if acid else "TSONOPOULOS"
        provider = create_second_virial_provider(correlation,list(pair),properties,allow_online=False,chemical_database=db)
    except (ValueError,RuntimeError,KeyError,OverflowError) as error:
        write_csv(Path(output)/("_".join(pair)+".csv"),[{"pair":" / ".join(pair),"status":str(error),"scope":"shared vapor initialization"}])
        return [],[{"pair":" / ".join(pair),"reason":str(error)}],[]
    for method in METHODS:
        # The generic existing BV wrapper is reused; only its table selection changes.
        filename = {"UNIFAC":"unifac_params.json","UNIFAC2":"unifac2/UNIFAC2rq.csv","UNIFDMD":"unifac_dmd.txt","UNIFM2":"unifac2/UNIFM2rq.xlsx","UNIFNIST":"nist_modified_unifac_params.json"}[method]
        cls = type("Benchmark"+method+"BV",(UNIFACBVThermodynamics,),{"unifac_variant":method,"default_unifac_data_filename":filename})
        try:
            model = cls(list(pair),db=db,second_virial_provider=provider,second_virial_correlation=correlation)
            if any("interaction parameters missing" in w for w in model.warnings):
                raise ValueError("incomplete UNIFAC group-interaction matrix")
        except (ValueError,RuntimeError,KeyError,OverflowError) as error:
            audit.append({"pair":" / ".join(pair),"method":method,"reason":str(error),"scope":"model initialization"})
            continue
        for measurement in rows:
            row = {"pair":" / ".join(pair),"method":method,"acid":acid,"vapor_correlation":correlation,**measurement}
            doi,t = row["doi"],row["T_K"]
            try:
                for c in pair:
                    curves.evaluate(doi,c,t)
                    if model.props[c].Tc is None or t >= model.props[c].Tc:
                        raise ValueError("measured temperature at/above common model Tc")
                row["psat_source1"],row["psat_source2"] = curves.used[doi,pair[0]],curves.used[doi,pair[1]]
                model.Psat = MethodType(lambda self,c,temperature,doi=doi:curves.evaluate(doi,c,temperature),model)
                # Source selection is in the cache identity through explicit clearing.
                for name in ("_psat_cache","_phi_sat_cache","_poynting_cache","_k_values_cache"):
                    getattr(model,name).clear()
                row.update(status='ok',**measured_state(model,measurement,reference_cache))
            except (ValueError,RuntimeError,KeyError,OverflowError,ZeroDivisionError) as error:
                row["status"] = str(error)
                audit.append({"pair":" / ".join(pair),"method":method,"doi":doi,"T_K":t,"reason":str(error),"scope":"point"})
            results.append(row)
        model_info.append({"pair":" / ".join(pair),"method":method,"vapor_correlation":correlation,"warnings":model.warnings,
                           "props":{c:{k:getattr(model.props[c],k) for k in ("Tc","Pc","Vc","omega")} for c in pair}})
    write_csv(Path(output)/("_".join(pair)+".csv"),results or audit)
    return results,audit,model_info


def aggregate_pair_scores(scores):
    result = {"pairs":len(scores),"points":sum(s["successes"] for s in scores)}
    for field in ('bubble_equation_AARD_percent','bubble_equation_ln_MAE',"ln_gamma_MAE"):
        values = [s[field] for s in scores if field in s]
        if values:
            result[field] = float(np.mean(values))
            result[field+"_median_pair"] = float(np.median(values))
            result[field+"_pairs"] = len(values)
    for field in ('bubble_equation_ln_MSE',"ln_gamma_MSE"):
        values = [s[field] for s in scores if field in s]
        if values:result[field.replace("MSE","RMS")] = float(np.sqrt(np.mean(values)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=5,choices=range(1,6))
    parser.add_argument("--summarize-dir",type=Path,help="Reaggregate saved measured-state gamma predictions without repeating numerical work")
    args = parser.parse_args()
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    point_dir=output/"pair_points";point_dir.mkdir()
    source=json.loads(args.source.read_text())
    input_files = [ROOT/'data'/name for name in ('perry_properties.json','chemicals.json','unifac_params.json','unifac_dmd.txt','nist_modified_unifac_params.json','unifac2/UNIFAC2rq.csv','unifac2/UNIFAC2int.csv','unifac2/UNIFM2rq.xlsx','unifac2/UNIFM2intA.xlsx','unifac2/UNIFM2intB.xlsx')]
    input_files += [ROOT/name for name in ('unifac.py','unifac_fragmenter.py','thermodynamics_models/gamma_phi.py','thermodynamics_models/activity.py','thermodynamics_models/second_virial.py','thermodynamics_models/phase_diagrams.py','scripts/activity_fitting/fit_fendu_2025_activity_parameters.py')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files}
    grouped=defaultdict(list)
    for r in source["rows"]:grouped[(r["cas1"],r["cas2"])].append(r)
    started=time.perf_counter()
    all_rows,audit,model_info=[],[],[]
    work=dict(grouped)
    old=None
    if args.summarize_dir:
        previous=args.summarize_dir.resolve()
        old=json.loads((previous/'summary.json').read_text())
        if hashlib.sha256(args.source.read_bytes()).hexdigest()!=old['source_sha256']:raise ValueError('Source changed since saved run')
        old_rows=list(csv.DictReader((previous/'points.csv').open()))
        for r in old_rows:
            r={k:v for k,v in r.items() if v!=''}
            for k,v in list(r.items()):
                if k=='acid':r[k]=v=='True'
                elif k in ('T_K','P_Pa','x1','y1') or k.startswith(('predicted_','inferred_','ln_gamma_error')):
                    r[k]=float(v)
            if r['status']=='ok' and all(k in r for k in ('y1','ln_gamma_error1','ln_gamma_error2')):
                total=r['y1']*math.exp(r['ln_gamma_error1'])+(1-r['y1'])*math.exp(r['ln_gamma_error2'])
                r.update(bubble_equation_relative_residual=total-1,bubble_equation_ln_residual=math.log(total))
            all_rows.append(r)
        audit=list(csv.DictReader((previous/'excluded.csv').open()))
        model_info=json.loads((previous/'model_info.json').read_text())
        work={}
        print('Reaggregating completed predictions; no numerical calculations repeated',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(evaluate_pair,pair,rows,source,point_dir):pair for pair,rows in work.items()}
        for i,future in enumerate(as_completed(futures),1):
            pair=futures[future]
            try:
                rows,errors,info=future.result()
            except Exception as error:
                rows,errors,info=[],[{"pair":" / ".join(pair),"reason":str(error),"scope":"worker"}],[]
            all_rows.extend(rows);audit.extend(errors);model_info.extend(info)
            print(f"Finished {i}/{len(grouped)} pairs: {' / '.join(pair)}, valid method-points={sum(r['status']=='ok' for r in rows)}",flush=True)
    # Compare identical successful point IDs across all five methods, rather
    # than silently evaluating each method on a different temperature domain.
    def point_key(r):return (r["doi"],r["cas1"],r["cas2"],r["T_K"],r["P_Pa"],r["x1"])
    success={m:{point_key(r) for r in all_rows if r["method"]==m and r["status"]=="ok"} for m in METHODS}
    common=set.intersection(*success.values())
    scores=[]
    for method in METHODS:
        for pair in grouped:
            subset=[dict(r) for r in all_rows if r["method"]==method and (r["cas1"],r["cas2"])==pair and point_key(r) in common]
            if subset:scores.append({"method":method,"pair":" / ".join(pair),"acid":subset[0]["acid"],**pair_metrics(subset)})
    for name,expected in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Input changed during run: {name}')
    summary={"protocol":"Direct equilibrium-equation evaluation at measured T/P/x/y. No bubble-pressure or temperature roots. Infer ln(gamma) only where vapor composition was measured; bubble-equation closure is a separate residual, not a solved pressure error. Shared source/Perry/Poling/Landolt/VDI pure Psat, Tsonopoulos for nonacids, association-aware HOC for acids, runtime phi_sat/Poynting. Equal weight per pair and identical measured points across methods. No liquid fitting.",
             "input_points":len(source["rows"]),"input_pairs":len(grouped),"common_points":len(common),
             "input_sha256":hashes,
             "grand_total":{m:aggregate_pair_scores([s for s in scores if s['method']==m]) for m in METHODS},
             "nonacid_total":{m:aggregate_pair_scores([s for s in scores if s['method']==m and not s['acid']]) for m in METHODS},
             "acid_total":{m:aggregate_pair_scores([s for s in scores if s['method']==m and s['acid']]) for m in METHODS},
             "coverage":{m:{"successful_points":len(success[m]),"unavailable_points":len(source['rows'])-len(success[m]),"recorded_point_failures":sum(r['method']==m and r['status']!='ok' for r in all_rows)} for m in METHODS},
             "elapsed_seconds":time.perf_counter()-started,"workers":args.workers,
             "source_sha256":hashlib.sha256(args.source.read_bytes()).hexdigest(),
             "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "versions":{n:version(n) for n in ('numpy','scipy','numba','rdkit','chemicals')},
             "limitations":"ln(gamma) inferred from measured y depends on common pure/vapor references. Pressure-only points cannot identify separate component gammas. Missing coverage and warnings are saved. ML training overlap unknown."}
    if old is not None:
        summary['completed_forward_checks']={k:old[k] for k in ('grand_total','nonacid_total','acid_total','common_temperature_points')}
        summary['cached_predictions_dir']=str(previous)
        summary['cached_predictions_sha256']=hashlib.sha256((previous/'points.csv').read_bytes()).hexdigest()
    write_csv(output/"points.csv",all_rows);write_csv(output/"pairs.csv",scores);write_csv(output/"excluded.csv",audit)
    (output/"model_info.json").write_text(json.dumps(model_info,indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
