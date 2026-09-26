#!/usr/bin/env python3
"""Rebuild offline sieve defaults from the retained NIST source records.

Only <=1 bar observations are fitted: excess loading is used as the
low-density approximation to absolute loading. Pure PR fugacities replace
pressure in the fit. This approximation and the fitted ranges are retained
in the runtime records and exposed by the unit's warnings.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thermodynamics import create_thermodynamics  # noqa: E402
from chemical_properties import ChemicalDatabase  # noqa: E402
from scripts.adsorption_fit_tools import select_fit, prediction, setting_from_fit  # noqa: E402


def main():
    source = json.loads((ROOT/'data/source/molecular_sieve_nist.json').read_text())
    corrections = json.loads((ROOT/'data/source/molecular_sieve_corrections.json').read_text())['record_corrections']
    names = {'Carbon Dioxide':'CO2', 'Methane':'CH4', 'Nitrogen':'N2',
             'Oxygen':'O2', 'Water':'water', 'Carbon monoxide':'CO',
             'Ethane':'ethane', 'Ethene':'ethylene', 'N-propane':'propane',
             'Propene':'propylene', 'Carbonyl sulfide':'carbonyl sulfide',
             'Hydrogen sulfide':'H2S', 'Acetylene':'acetylene',
             'Sulfur Hexafluoride':'sulfur hexafluoride', 'Formaldehyde':'formaldehyde',
             'Isobutane':'isobutane', '1-butene':'1-butene',
             'Argon':'Ar', 'Neon':'Ne', 'Helium':'He', 'Ammonia':'NH3'}
    groups = {}
    excluded_records = dict(source.get('excluded_records', {}))
    ammonia = json.loads((ROOT/'data/source/molecular_sieve_ammonia.json').read_text())
    ammonia_record = {
        'filename':'Sakata1994.Figure6.Z1.regenerated', 'DOI':ammonia['DOI'],
        'adsorbent':{'name':'Zeolite 3A'}, 'adsorbates':[{'name':'Ammonia'}],
        'category':'exp', 'adsorptionUnits':'cm3(STP)/g','pressureUnits':'bar',
        'temperature':ammonia['temperature_K'], 'source_provider':ammonia['source_provider'],
        'source_file':'source/molecular_sieve_ammonia.json',
        'notes':ammonia['limitations'],
        'isotherm_data':[{'pressure':p*1.01325/760, 'species_data':[{'adsorption':q}]}
                         for p,q in zip(ammonia['pressure_Torr'],ammonia['loading_cm3_STP_per_g'])],
    }
    for record in [*source['records'],ammonia_record]:
        if record['filename'] in excluded_records:
            continue
        assert record['adsorptionUnits'] in {'mmol/g','cm3(STP)/g','wt%'}
        assert record['pressureUnits'] == 'bar'
        assert record['category'] == 'exp' and len(record['adsorbates']) == 1
        c = names[record['adsorbates'][0]['name']]
        sieve = record['adsorbent']['name'].split()[-1]
        points = sorted((float(p['pressure']),float(p['species_data'][0]['adsorption']))
                        for p in record['isotherm_data'] if 0<float(p['pressure'])<=1)
        if not points:
            excluded_records[record['filename']] = 'No observations within the <=1 bar fitting domain; high-pressure excess/absolute conversion not established.'
            continue
        if points:
            running_max = np.maximum.accumulate([q for _,q in points])
            if max(running_max-np.array([q for _,q in points])) > .1*max(running_max):
                excluded_records[record['filename']] = 'Loading decreases by more than 10% of its maximum as pressure rises; not a usable monotonic equilibrium branch.'
                print(f'SKIP {record["filename"]}: strongly nonmonotonic data')
                continue
        groups.setdefault((sieve,c), []).append(record)
    output = {'source':'source/molecular_sieve_nist.json', 'commit':source['commit'],
              'sieves':{}, 'species':{}, 'excluded_pairs':[], 'excluded_records':excluded_records}
    db = ChemicalDatabase(enable_online=False)
    for (sieve,c), records in sorted(groups.items()):
        if db.get(c,fetch_online=False) is None:
            output['excluded_pairs'].append({'sieve':sieve,'component':c,'reason':'No local pure-component properties for conversion to fugacity'})
            print(f'SKIP {sieve}/{c}: no local pure-component properties')
            continue
        thermo = create_thermodynamics([c], 'PR', db=db)
        props = thermo.props[c]
        output['species'][c] = {'CAS':props.CAS,
            'aliases':sorted({c.lower(),props.name.lower(),*(r['adsorbates'][0]['name'].lower() for r in records)})}
        data = []
        used = []
        for record in records:
            rows = []
            for point in sorted(record['isotherm_data'], key=lambda value:value['pressure']):
                p = float(point['pressure'])
                q = float(point['species_data'][0]['adsorption'])
                if record['adsorptionUnits']=='cm3(STP)/g':
                    q /= 22.41396954  # 273.15 K, 1 atm; cm3(STP)/g -> mol/kg.
                elif record['adsorptionUnits']=='wt%':
                    q *= 10/props.MW
                if 0 < p <= 1 and q > 0:
                    t = float(corrections.get(record['filename'], {}).get('temperature_K', record['temperature']))
                    phi = thermo.fugacity_coefficients(t,p,{c:1.},'vapor')[c]
                    rows.append((p*phi,t,q,p,len(rows)%3))
            if rows:
                used.append(record['filename'])
                data.extend(rows)
        a = np.array(data)
        f,t,q,p,fold = a.T
        if len(q) < 4 or min(t)<=0:
            raise RuntimeError(f'Insufficient low-pressure observations for {sieve}/{c}')
        thermal = len(set(t)) > 1
        baseline = 'dual_site_langmuir' if len(q)>=10 else 'langmuir'
        form, fitted, candidates = select_fit(f,t,q,fold,baseline)
        predicted = prediction(fitted.x,form,f,t,thermal)
        entry = {**setting_from_fit(form,fitted,thermal),
                 'fit_selection': {'method':candidates[form]['validation_method'],
                     'baseline_model':baseline,'selected_model':form,'candidates':candidates},
                 'temperature_corrections':{r:corrections[r] for r in used if r in corrections},
                 'low_pressure_limit':'non-Henry empirical extrapolation' if form=='sips' else 'Henry',
                 'T_min':float(t.min()),'T_max':float(t.max()),
                 'f_min':float(f.min()),'f_max':float(f.max()),
                 'pressure_max_bar':float(p.max()),
                 'source_records':used,'DOIs':sorted({r['DOI'] for r in records}),
                 'source_provider':records[0].get('source_provider','NIST ISODB'),
                 'source_file':records[0].get('source_file','source/molecular_sieve_nist.json'),
                 'notes':records[0].get('notes',''),
                 'points':len(q),'rmse_mol_kg':float(np.sqrt(np.mean((predicted-q)**2))),
                 'mean_absolute_relative_error':float(np.mean(abs(predicted-q)/q)),
                 'loading_basis':'absolute' if all(r.get('isotherm_type')=='absolute' for r in records) else 'low-pressure reported loading approximation to absolute',
                 'fugacity_basis':'pure-component PR fugacity',
                 'temperature_fit':'vanthoff' if len(set(t))>1 else 'single temperature; heats unidentifiable'}
        if entry['mean_absolute_relative_error'] > .25:
            output['excluded_pairs'].append({'sieve':sieve,'component':c,
                'reason':'Mean absolute relative fit error exceeds 25%; source requires further assessment',
                'fit_error':entry['mean_absolute_relative_error'],'source_records':used})
            print(f'SKIP {sieve}/{c}: MARE={entry["mean_absolute_relative_error"]:.2%}')
            continue
        output['sieves'].setdefault(sieve,{})[c] = entry
        print(f'{sieve:3} {c:5} {len(q):3} points; {form}; RMSE={entry["rmse_mol_kg"]:.5g} mol/kg; MARE={entry["mean_absolute_relative_error"]:.2%}')
    path = ROOT/'data/molecular_sieve_isotherms.json'
    path.write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
