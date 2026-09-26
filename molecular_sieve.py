"""Isothermal equilibrium molecular-sieve unit with automatic coadsorption."""

import math

import numpy as np
from scipy.optimize import least_squares

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .adsorption_models import GSTA_3A_WATER, PureIsotherm, iast, iast_enthalpy, iast_isosteric_heats, normalize_sieve, positive, sieve_database
    from .unit_operations_base import UnitOperation, UnitOperationError, UnitResult
    from .unit_conversions import mass_flow_to_kg_per_hour
else:
    from adsorption_models import GSTA_3A_WATER, PureIsotherm, iast, iast_enthalpy, iast_isosteric_heats, normalize_sieve, positive, sieve_database
    from unit_operations_base import UnitOperation, UnitOperationError, UnitResult
    from unit_conversions import mass_flow_to_kg_per_hour


# Approximate kinetic diameters, used ONLY to warn about missing curves.
# A measured/configured isotherm takes precedence over a nominal pore cutoff.
# References and limitations: docs/molecular_sieves.md.
SPECIES = {
    'water': ('7732-18-5', 2.65, ('water', 'h2o')),
    'CO2': ('124-38-9', 3.30, ('co2', 'carbon dioxide')),
    'N2': ('7727-37-9', 3.64, ('n2', 'nitrogen')),
    'O2': ('7782-44-7', 3.46, ('o2', 'oxygen')),
    'CH4': ('74-82-8', 3.80, ('ch4', 'methane')),
    'H2': ('1333-74-0', 2.89, ('h2', 'hydrogen')),
    'He': ('7440-59-7', 2.60, ('he', 'helium')),
    'Ne': ('7440-01-9', 2.75, ('ne', 'neon')),
    'Ar': ('7440-37-1', 3.40, ('ar', 'argon')),
    'CO': ('630-08-0', 3.76, ('co', 'carbon monoxide')),
    'NH3': ('7664-41-7', 2.60, ('nh3', 'ammonia')),
    'methanol': ('67-56-1', 3.60, ('methanol', 'ch3oh')),
    'ethanol': ('64-17-5', 4.30, ('ethanol', 'c2h5oh')),
}
PORES = {'3A':3.0, '4A':4.0, '5A':5.0, '13X':7.4}


class MolecularSieveDryer(UnitOperation):
    """One equilibrium contact with dry/regenerated sieve, in kg/h.

    Every available isotherm participates in IAST. The adsorbate outlet is
    a material accounting stream, not a separate equilibrium fluid phase.
    """

    ADSORBENT_MASS_NAMES = (
        'adsorbent_mass_flow', 'sieve_mass_flow', 'molecular_sieve_mass_flow',
        'adsorbent_flow', 'sieve_flow', 'adsorbent_mass', 'sieve_mass',
    )

    def _identity(self, component):
        prop = self.thermo.props.get(component)
        cas = getattr(prop, 'CAS', '')
        names = {component.lower(), str(getattr(prop, 'name', '')).lower()}
        for key, record in sieve_database().get('species', {}).items():
            if (cas and cas == record.get('CAS')) or names.intersection(record['aliases']):
                return key
        for key, (known_cas, _, aliases) in SPECIES.items():
            if cas == known_cas or names.intersection(aliases):
                return key
        return component

    def _mapping(self, name):
        value = self.get_param(name, {})
        if not isinstance(value, dict):
            raise ValueError(f'{name} must be a mapping keyed by component')
        return value

    def solve(self, inlets):
        try:
            return self._solve(inlets)
        except UnitOperationError:
            raise
        except (ValueError, TypeError, ArithmeticError) as error:
            raise UnitOperationError(f"MolecularSieveDryer '{self.unit_id}': {error}") from error

    def _solve(self, inlets):
        if len(inlets) != 1:
            raise ValueError('requires exactly one feed inlet')
        inlet = next(iter(inlets.values()))
        positive(inlet.F, 'Feed flow')
        if getattr(inlet, 'liquid2_fraction', 0.) > 1e-8:
            raise ValueError('requires liquid-liquid feeds to be separated before adsorption')
        flash_product = 1e-8 < inlet.vapor_fraction < 1-1e-8
        phase = 'vapor' if inlet.vapor_fraction >= .5 else 'liquid'
        sieve = normalize_sieve(self.get_param('sieve_type', self.get_param('molecular_sieve','3A')))
        warnings = []
        components = list(inlet.composition)
        identities = {c:self._identity(c) for c in components}
        water = self.get_param('water_component')
        if water is None:
            water = next((c for c in components if identities[c]=='water'), None)
        elif water not in components:
            raise ValueError('water_component is not in the feed')
        target_component = self.get_param('target_component', water)
        specs = [name for name in (*self.ADSORBENT_MASS_NAMES,
                 'target_water_mole_fraction','target_mole_fraction','removal_fraction')
                 if self.get_param(name) is not None]
        if len(specs) != 1:
            raise ValueError('requires exactly one sieve flow or sizing target; specifications are mutually exclusive')
        spec = specs[0]
        fixed_rate = spec in self.ADSORBENT_MASS_NAMES
        target = None
        if fixed_rate:
            rate = positive(self.get_param(spec), 'Adsorbent mass flow')
            rate = mass_flow_to_kg_per_hour(rate, self.get_param_unit(spec))
        else:
            if spec == 'target_water_mole_fraction':
                target_component = water
            if target_component not in components:
                raise ValueError('sizing requires target_component (or a water component) in the feed')
            target = positive(self.get_param(spec), spec, zero=True)
            if spec == 'removal_fraction':
                if target > 1:
                    raise ValueError('removal_fraction must be between 0 and 1')
                target = inlet.F*inlet.composition[target_component]*(1-target)
            elif target >= 1:
                raise ValueError('target mole fraction must be < 1')
            if target == 0 and inlet.composition[target_component] > 0:
                raise ValueError('exactly zero residual adsorbate is thermodynamically impossible at finite sieve flow')

        defaults = sieve_database()['sieves'].get(sieve,{})
        overrides = self._mapping('isotherms')
        sizes = self._mapping('kinetic_diameters')
        for name in (*overrides, *sizes):
            if name not in components and name not in identities.values():
                raise ValueError(f'Unknown feed component {name!r} in adsorption settings')
        pore = self.get_param('pore_diameter',PORES.get(sieve))
        if pore is not None:
            pore = positive(pore, 'pore_diameter [angstrom]')
        settings = {}
        for c in components:
            ident = identities[c]
            setting = overrides.get(c,overrides.get(ident,defaults.get(ident)))
            if setting is not None and c not in overrides and ident not in overrides:
                if (setting.get('T_min') == setting.get('T_max')
                        and setting.get('T_min') is not None
                        and abs(inlet.T-setting['T_min']) > 1.):
                    warnings.append(f'{c}/{sieve}: only a {setting["T_min"]:g} K isotherm is available; it is not valid at {inlet.T:g} K without a temperature dependence. Supply a custom isotherm; this component passes through.')
                    setting = None
            if setting is None and sieve == '3A' and ident == 'water':
                setting = GSTA_3A_WATER
            if setting is not None:
                settings[c] = setting
            elif inlet.composition[c] > 0:
                diameter = sizes.get(c,sizes.get(ident,SPECIES.get(ident,('',None,()))[1]))
                if diameter is not None:
                    diameter = positive(diameter, f'{c} kinetic diameter')
                if diameter is None or pore is None:
                    warnings.append(f'{c}: no isotherm for {sieve}; molecular size or sieve pore size is unknown, so adsorption cannot be assessed.')
                elif diameter <= pore:
                    warnings.append(f'{c}: fits the nominal {sieve} pore ({diameter:g} <= {pore:g} angstrom), but its isotherm is missing; it passes through unadsorbed.')
        models = {c:PureIsotherm(v,inlet.T) for c,v in settings.items()}
        if not fixed_rate and target_component not in models:
            raise ValueError(f'No valid {sieve} isotherm for sizing component {target_component}')

        initial = self._mapping('initial_loadings')
        for c in initial:
            if c not in models:
                raise ValueError(f'Initial loading specified without an isotherm for {c}')
        scalar_initial = self.get_param('initial_loading_kg_per_kg',self.get_param('initial_loading'))
        if scalar_initial is not None:
            if water not in models or water in initial:
                raise ValueError('Scalar initial_loading requires water and cannot duplicate initial_loadings')
            initial = dict(initial, **{water:scalar_initial})
        molecular_weights = {}
        for c in models:
            molecular_weights[c] = positive(getattr(self.thermo.props.get(c),'MW',None),f'{c} molecular weight')
        initial_mol = {c:positive(initial.get(c,0.),f'{c} initial loading',zero=True)*1000/molecular_weights[c] for c in models}
        for c,m in models.items():
            capacity = (getattr(m,'qmax',float('inf')) + getattr(m,'qmax2',0.)) if m.model not in {'henry','freundlich','custom'} else float('inf')
            if initial_mol[c] > capacity+1e-10:
                raise ValueError(f'{c} initial loading exceeds maximum loading')

        feed = {c:inlet.F*inlet.composition[c] for c in components}
        active = [c for c in models if feed[c] > 0 or initial_mol[c] > 0]
        if any(feed[c] == 0 and initial_mol[c] > 0 for c in active):
            raise ValueError('Loaded sieve would desorb a component absent from the feed; this adsorption unit requires regenerated sieve')

        def equilibrium(flows):
            total = sum(flows.values())
            if total <= 0:
                raise ValueError('Adsorption removed the entire stream')
            composition = {c:v/total for c,v in flows.items()}
            contact_phase = phase
            if flash_product:
                state = self.thermo.calculate_state(inlet.T,inlet.P,total,composition,include=())
                if state.liquid2_fraction > 1e-8:
                    raise ValueError('adsorption contact forms a second liquid phase; separate liquid-liquid phases before adsorption')
                contact_phase = 'vapor' if state.vapor_fraction >= .5 else 'liquid'
                composition = (state.y if contact_phase=='vapor' else state.x) or composition
            fugacities = self.thermo.component_activities(inlet.T,inlet.P,composition,contact_phase)
            # component_activities uses a 1-bar reference, so numerical values
            # are fugacities in bar. Use the shared thermodynamic backend.
            return iast(models,fugacities), fugacities

        (qin,_,_),fin = equilibrium(feed)
        for c,m in models.items():
            m.validate_range(max(fin.get(c,0.),inlet.P))
        if not fixed_rate:
            current = feed[target_component] if spec=='removal_fraction' else inlet.composition[target_component]
            if target >= current:
                rate = 0.
                flows = dict(feed)
                q,pi,f0 = qin,0.,{}
                fugacities = fin
            elif len([v for v in feed.values() if v>0]) == 1 and spec!='removal_fraction':
                raise ValueError('cannot change mole fraction without a carrier component')
            else:
                rate = max(inlet.F*1000/max(sum(qin.values()),1e-15),1e-12)
                flows = None
        else:
            flows = None
        if not active:
            flows = dict(feed)
            q,pi,f0 = qin,0.,{}
            fugacities = fin

        residual_error = 0.
        if flows is None:
            sizing = not fixed_rate
            def unpack(values):
                mrate = math.exp(values[-1]) if sizing else rate
                output = dict(feed)
                output.update({c:feed[c]*math.exp(-u) for c,u in zip(active,values)})
                return output,mrate
            def residual(values):
                output,mrate = unpack(values)
                (loading,_,_),_ = equilibrium(output)
                errors = [math.log((output[c]+mrate*loading[c]/1000)/(feed[c]+mrate*initial_mol[c]/1000)) for c in active]
                if sizing:
                    achieved = output[target_component]
                    if spec != 'removal_fraction':
                        achieved /= sum(output.values())
                    errors.append(math.log(achieved/target))
                return errors
            guess = np.array([max(.01,math.log1p(rate*qin[c]/(1000*feed[c]))) for c in active])
            if sizing:
                guess = np.r_[guess,math.log(rate)]
            # Allow trial desorption so a nearly unadsorbed species does not
            # produce a false projected-gradient convergence at u=0.
            lower = [-30.]*len(active)+([-60.] if sizing else [])
            upper = [690.]*len(active)+([100.] if sizing else [])
            answer = least_squares(residual,guess,bounds=(lower,upper),
                ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=400)
            residual_error = max(abs(np.array(residual(answer.x))))
            if residual_error > 2e-9:
                raise ValueError(f'Coupled adsorption balance did not converge (relative residual {residual_error:.3g}); target may be thermodynamically impossible or require desorption of loaded sieve')
            flows,rate = unpack(answer.x)
            if any(flows[c] > feed[c]+max(1e-12,feed[c]*1e-9) for c in active):
                raise ValueError('Supplied sieve is too wet/loaded: outlet equilibrium requires desorption; regenerate it before this adsorption-only unit')
            (q,pi,f0),fugacities = equilibrium(flows)
            for c,m in models.items():
                m.validate_range(f0.get(c,fugacities.get(c,0.)))

        removed = {c:max(0.,feed[c]-flows[c]) for c in components}
        total = sum(flows.values())
        if total < inlet.F*1e-14:
            raise ValueError('Adsorption removed the entire stream; no finite product equilibrium remains')
        product = self.thermo.calculate_state(inlet.T,inlet.P,total,
            {c:v/total for c,v in flows.items() if v>0},phase=None if flash_product else phase,flash=flash_product)
        adsorbate_total = sum(removed.values())
        adsorbate_comp = {c:v/adsorbate_total for c,v in removed.items() if v>0} if adsorbate_total>0 else dict(inlet.composition)
        # Retain the historical liquid-water accounting state for the 3A curve.
        # Other components use the feed phase, avoiding forced liquid permanent gases.
        accounting_phase = 'liquid' if len(models)==1 and water in models and models[water].model=='gsta_3a_water' else phase
        adsorbate = self.thermo.calculate_state(inlet.T,inlet.P,max(adsorbate_total,1e-12),adsorbate_comp,phase=accounting_phase,flash=False)
        adsorbate.F = adsorbate_total
        duty = product.F*(product.H or 0.)+adsorbate.F*(adsorbate.H or 0.)-inlet.F*(inlet.H or 0.)
        final = {c:initial_mol[c]*molecular_weights[c]/1000 + removed[c]*molecular_weights[c]/rate if rate else initial_mol[c]*molecular_weights[c]/1000 for c in models}
        performance = {
            'sieve_type':sieve,'mode':'equilibrium_isothermal_pseudo_continuous',
            'competitive_model':'IAST','adsorbed_components':list(models),
            'adsorbent_mass_flow_kg_h':rate,'removed_kmol_h':removed,
            'initial_loadings_kg_per_kg':{c:initial_mol[c]*molecular_weights[c]/1000 for c in models},
            'final_loadings_kg_per_kg':final,
            'equilibrium_loadings_mol_per_kg':q,
            'fugacities_bar':{c:fugacities.get(c,0.) for c in models},
            'iast_fictitious_fugacities_bar':f0,'spreading_potential_mol_per_kg':pi,
            'equilibrium_balance_relative_residual':residual_error,
            'stream_enthalpy_duty_kJ_h':duty,
            'adsorbate_stream_basis':'fluid material-accounting stream; excludes adsorbent enthalpy',
            'isotherm_sources':{c:settings[c].get('source_records',['custom' if c in overrides or identities[c] in overrides else 'existing 3A GSTA']) for c in models},
            'isotherm_source_providers':{c:settings[c].get('source_provider','custom or retained GSTA') for c in models},
        }
        if water in components:
            performance.update(water_component=water,water_removed_kmol_h=removed[water],
                water_removed_kg_h=removed[water]*getattr(self.thermo.props[water],'MW'),
                water_removal_fraction=removed[water]/max(feed[water],1e-300),
                product_water_mole_fraction=product.composition.get(water,0.))
        if water in models:
            performance.update(initial_loading_kg_water_per_kg_sieve=initial_mol[water]*molecular_weights[water]/1000,
                final_loading_kg_water_per_kg_sieve=final[water],
                equilibrium_loading_kg_water_per_kg_sieve=q[water]*molecular_weights[water]/1000,
                adsorption_reduced_pressure=fugacities.get(water,0.),
                driver_basis='vapor_fugacity_bar' if phase=='vapor' else 'liquid_water_activity',
                water_activity_or_fugacity_driver=fugacities.get(water,0.) if phase=='vapor' else fugacities.get(water,0.)/self.thermo.Psat(water,inlet.T))
        missing_heats = [c for c,m in models.items()
                         if (q[c]>0 or initial_mol[c]>0) and not m.enthalpy_available]
        correction = 0.
        performance['adsorption_enthalpy_missing_components'] = missing_heats
        if rate == 0 or not active:
            performance.update(adsorption_heat_release_kJ_h=0., bed_heat_duty_kJ_h=duty,
                               heat_duty_basis='physical isothermal bed; no adsorption')
        elif missing_heats:
            performance.update(bed_heat_duty_kJ_h=None,
                heat_duty_basis='fluid accounting only; adsorption enthalpy unavailable')
            warnings.append('Physical bed heat duty is unavailable: adsorption enthalpy is not identifiable for '
                + ', '.join(missing_heats) + '. Supply temperature-dependent isotherms or explicit heat parameters. Reported heat_duty includes fluid accounting only.')
        else:
            initial_h = iast_enthalpy(models,initial_mol)/1000  # kJ/kg
            final_h = iast_enthalpy(models,q,f0)/1000
            excess_change = rate*(final_h-initial_h)
            # Pure ideal-gas reference enthalpies are kJ/mol. Net kmol/h
            # uptake supplies the reference inventory change without needing
            # the arbitrary absolute enthalpy of the unchanged dry solid.
            gas_reference = sum(removed[c]*1000*self.thermo.enthalpy_ideal_gas(c,inlet.T)
                                for c in models)
            adsorbed_change = gas_reference+excess_change
            correction = adsorbed_change-adsorbate.F*(adsorbate.H or 0.)
            duty += correction
            differential = {c:v/1000 for c,v in iast_isosteric_heats(models,q).items()}
            performance.update(
                bed_heat_duty_kJ_h=duty,heat_duty_basis='physical isothermal bed',
                adsorption_heat_release_kJ_h=-excess_change,heat_release_kJ_h=-excess_change,
                adsorption_heat_reference='ideal gases at contact temperature; finite inventory enthalpy change',
                initial_adsorption_excess_enthalpy_kJ_per_kg=initial_h,
                final_adsorption_excess_enthalpy_kJ_per_kg=final_h,
                adsorbed_inventory_enthalpy_change_kJ_h=adsorbed_change,
                unrepresented_enthalpy_change_kJ_h=correction,
                isosteric_heats_kJ_per_mol=differential,
            )
            if water in models and len(models)==1:
                performance['heat_of_adsorption_kJ_per_mol_water'] = (
                    -excess_change/(removed[water]*1000) if removed[water]>0 else differential[water]
                )
        for c,setting in settings.items():
            if 'source_records' in setting and setting.get('loading_basis')!='absolute':
                provider = setting.get('source_provider','NIST ISODB')
                warnings.append(f'{c}/{sieve}: {provider} default uses low-pressure reported loading as an approximation to absolute adsorption; adsorbent batches may differ.')
            if setting.get('notes'):
                warnings.append(f'{c}/{sieve}: {setting["notes"]}')
            if inlet.T < setting.get('T_min',0.) or inlet.T > setting.get('T_max',math.inf):
                warnings.append(f'{c}/{sieve}: temperature {inlet.T:g} K is outside the isotherm data range [{setting["T_min"]:g}, {setting["T_max"]:g}] K.')
            if f0.get(c,0.) > setting.get('f_max',math.inf) or (0<f0.get(c,0.)<setting.get('f_min',0.)):
                warnings.append(f'{c}/{sieve}: IAST fictitious pure fugacity {f0[c]:g} bar extrapolates beyond the measured isotherm range.')
                if setting.get('low_pressure_limit')=='non-Henry empirical extrapolation':
                    warnings.append(f'{c}/{sieve}: the selected Sips fit has no finite nonzero Henry limit; extrapolated dilute adsorption and enthalpy are empirical estimates.')
        return UnitResult(outlet_streams={'product':product,'adsorbate':adsorbate},
                          heat_duty=duty,performance=performance,warnings=warnings,
                          unrepresented_enthalpy_change=correction)
