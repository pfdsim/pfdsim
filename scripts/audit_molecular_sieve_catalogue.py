#!/usr/bin/env python3
"""Report selected forms, thermal coverage, and independent 4A/CH4 checks."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adsorption_models import PureIsotherm, iast_isosteric_heats  # noqa: E402
from chemical_properties import ChemicalDatabase  # noqa: E402
from thermodynamics import create_thermodynamics  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--output',type=Path,default=Path('/tmp/pfdsim-isotherm-selection-audit.json'))
    args=parser.parse_args()
    catalogue=json.loads((ROOT/'data/molecular_sieve_isotherms.json').read_text())
    baseline=json.loads(args.baseline.read_text()) if args.baseline else None
    entries=[]
    for sieve,curves in catalogue['sieves'].items():
        for c,setting in curves.items():
            old=baseline['sieves'][sieve][c] if baseline else None
            row={'sieve':sieve,'component':c,'model':setting['model'],
                 'train_mare':setting['mean_absolute_relative_error'],
                 'validation_mare':setting['fit_selection']['candidates'][setting['model']]['validation_mare'],
                 'temperature_range_K':[setting['T_min'],setting['T_max']],
                 'temperature_dependent':setting['temperature_fit']=='vanthoff'}
            if old:
                row.update(previous_model=old['model'],previous_train_mare=old['mean_absolute_relative_error'])
            entries.append(row)
    setting=catalogue['sieves']['4A']['CH4']
    thermo=create_thermodynamics(['CH4'],'PR',db=ChemicalDatabase(enable_online=False))
    source=json.loads((ROOT/'data/source/methane_4a_external_validation.json').read_text())
    checks=[]
    for record in source['records']:
        point=min(record['isotherm_data'],key=lambda v:v['pressure'])
        p,t=point['pressure'],record['temperature']
        f=p*thermo.fugacity_coefficients(t,p,{'CH4':1.},'vapor')['CH4']
        iso=PureIsotherm(setting,t)
        predicted=iso.loading(f)
        observed=point['species_data'][0]['adsorption']
        checks.append({'source_record':record['filename'],'temperature_K':t,'pressure_bar':p,
            'observed_mol_kg':observed,'predicted_mol_kg':predicted,
            'relative_error':predicted/observed-1,
            'temperature_extrapolation':not setting['T_min']<=t<=setting['T_max'],
            'fugacity_extrapolation':not setting['f_min']<=f<=setting['f_max'],
            'caveat':'Different adsorbent batch; reported excess loading approximates absolute at low pressure. Not fitted.'})
    heats=[]
    for temperature in [272.990,297.991,322.989]:
        iso=PureIsotherm(setting,temperature)
        q=iso.loading(.5)
        heat=iast_isosteric_heats({'CH4':iso},{'CH4':q})['CH4']/1000
        heats.append({'temperature_K':temperature,'fugacity_bar':.5,'loading_mol_kg':q,'Qst_kJ_mol':heat})
    report={'entries':entries,'fitted_pairs':len(entries),
            'thermal_fitted_pairs':sum(x['temperature_dependent'] for x in entries),
            'additional_retained_curve':'3A water GSTA, temperature dependent',
            'methane_4a_external_validation':checks,'methane_4a_isosteric_heats':heats}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f"{len(entries)} fitted pairs, {report['thermal_fitted_pairs']} temperature-dependent, plus retained 3A water")
    for row in entries:
        if baseline and (row['model']!=row['previous_model'] or (row['sieve'],row['component'])==('4A','CH4')):
            print(row['sieve'],row['component'],row['previous_model'],'->',row['model'],
                  f"fit {row['previous_train_mare']:.2%} -> {row['train_mare']:.2%}; CV {row['validation_mare']:.2%}")
    for row in checks:print('External methane:',row)
    for row in heats:print('Methane heat:',row)


if __name__=='__main__':main()
