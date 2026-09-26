#!/usr/bin/env python3
"""Compare isotherm forms on the worst retained NIST fits, without changing defaults.

Deterministic interleaved three-fold validation within each source curve.
Reports training and held-out relative errors using the builder's selector,
including audited source corrections and small-dataset eligibility checks.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemical_properties import ChemicalDatabase  # noqa: E402
from thermodynamics import create_thermodynamics  # noqa: E402
from scripts.adsorption_fit_tools import select_fit  # noqa: E402



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('/tmp/pfdsim-adsorption-fit-comparison.json'))
    parser.add_argument('--count',type=int,default=10)
    args=parser.parse_args()
    catalogue=json.loads((ROOT/'data/molecular_sieve_isotherms.json').read_text())
    corrections=json.loads((ROOT/'data/source/molecular_sieve_corrections.json').read_text())['record_corrections']
    records={r['filename']:r for r in json.loads((ROOT/'data/source/molecular_sieve_nist.json').read_text())['records']}
    worst=sorted([(v['mean_absolute_relative_error'],s,c,v) for s,d in catalogue['sieves'].items() for c,v in d.items()
                  if all(r in records for r in v['source_records'])],reverse=True)[:args.count]
    db=ChemicalDatabase(enable_online=False)
    report=[]
    for old,s,c,entry in worst:
        thermo=create_thermodynamics([c],'PR',db=db)
        rows=[]
        for record_id in entry['source_records']:
            record=records[record_id]
            index=0
            for point in sorted(record['isotherm_data'],key=lambda p:p['pressure']):
                p,q=point['pressure'],point['species_data'][0]['adsorption']
                if not (0<p<=1 and q>0):continue
                units=record['adsorptionUnits']
                if units=='cm3(STP)/g':q/=22.41396954
                elif units=='wt%':q*=10/thermo.props[c].MW
                elif units!='mmol/g':raise ValueError(units)
                t=corrections.get(record_id,{}).get('temperature_K',record['temperature'])
                f=p*thermo.fugacity_coefficients(t,p,{c:1.},'vapor')[c]
                rows.append((f,t,q,index%3,record_id));index+=1
        f,t,q,fold=np.array([r[:4] for r in rows]).T
        chosen,_,results=select_fit(f,t,q,fold,entry['model'])
        report.append({'sieve':s,'component':c,'points':len(q),'old_mare':old,'selected_model':chosen,'models':results})
        print(s,c,'old',round(old*100,2),'train/CV',
              {k:(round(v['train_mare']*100,2),round(v['validation_mare']*100,2)) if v['eligible'] else v['reason'] for k,v in results.items()},flush=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
