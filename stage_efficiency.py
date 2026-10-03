"""Shared vapor Murphree variables, equations and derivatives for column models."""

import math
import numpy as np

if __package__ and __package__.split('.',1)[0] == 'pfdsim':
    from .distillation_specifications import stage_efficiency_specification
    from .unit_operations_base import UnitOperationError
else:
    from distillation_specifications import stage_efficiency_specification
    from unit_operations_base import UnitOperationError


class VaporStageEfficiencies:
    """Eliminate equilibrium-stage vapors; retain unknowns only where E != 1."""

    def __init__(self,unit,components,stages,feed_specs,column_start,row_start):
        self.unit,self.components = unit,tuple(components)
        try:
            self.values = stage_efficiency_specification(unit.params,stages)
        except ValueError as error:
            raise UnitOperationError(f"{unit.unit_id}: {error}") from error
        self.active = tuple(j for j in range(1,stages-1) if self.values[j] != 1.)
        width = len(self.components)-1
        self.columns = {j:tuple(range(column_start+i*width,column_start+(i+1)*width))
                        for i,j in enumerate(self.active)}
        self.rows = {j:tuple(range(row_start+i*width,row_start+(i+1)*width))
                     for i,j in enumerate(self.active)}
        self.extra_size = len(self.active)*width
        self.feed_flow = np.zeros(stages)
        self.feed_moles = [dict.fromkeys(self.components,0.) for _ in range(stages)]
        if not self.active:
            return
        for feed in feed_specs:
            amount = feed['F']*feed.get('vapor_fraction',0.)
            if amount <= 0:
                continue
            y = feed.get('vapor_composition')
            if y is None:
                if feed['vapor_fraction'] < 1.-1e-12:
                    raise UnitOperationError(
                        f"{unit.unit_id}: a two-phase feed requires its actual vapor composition for stage efficiencies")
                y = feed['z']
            y = self.normalize(y)
            j = int(feed['stage'])
            self.feed_flow[j] += amount
            for c in self.components:
                self.feed_moles[j][c] += amount*y[c]

    def normalize(self,composition):
        values = {c:float(composition.get(c,0.)) for c in self.components}
        if any(not math.isfinite(v) or v < 0 for v in values.values()) or sum(values.values()) <= 0:
            raise UnitOperationError(f'{self.unit.unit_id}: invalid vapor composition')
        total = sum(values.values())
        return {c:v/total for c,v in values.items()}

    def decode(self,vector,j):
        columns = self.columns.get(j)
        if columns is None:
            return None
        logits = np.asarray([*(vector[col] for col in columns),0.],dtype=float)
        weights = np.exp(logits-logits.max())
        weights /= weights.sum()
        return dict(zip(self.components,map(float,weights)))

    def pack(self,vector,compositions):
        if self.active and compositions is None:
            raise UnitOperationError(f'{self.unit.unit_id}: efficiency initialization requires vapor compositions')
        for j in self.active:
            y = self.normalize(compositions[j])
            last = max(y[self.components[-1]],1e-14)
            vector[list(self.columns[j])] = [math.log(max(y[c],1e-14)/last) for c in self.components[:-1]]

    def actual_properties(self,props,j,T,P,y):
        """Preserve the equilibrium reference, evaluate calorics for actual gas."""
        reference = props.get('equilibrium_y',props['y'])
        result = dict(props,equilibrium_y=reference)
        if y is not None:
            result['y'] = y
            result['hV'] = self.unit.thermo.mixture_enthalpy(y,float(T),1.,P=float(P))
        return result

    def incoming(self,j,below_flow,below_y):
        total = below_flow+self.feed_flow[j]
        return ({c:(below_flow*below_y[c]+self.feed_moles[j][c])/total for c in self.components},
                below_flow/total)

    def residuals(self,props,vapor_flows,*,all_components=False):
        values = []
        comps = self.components if all_components else self.components[:-1]
        for j in self.active:
            yin,_ = self.incoming(j,vapor_flows[j+1],props[j+1]['y'])
            values.extend(props[j]['y'][c]-yin[c]-self.values[j]*(props[j]['equilibrium_y'][c]-yin[c])
                          for c in comps)
        return values

    def mark_sparsity(self,matrix,thermal_columns):
        for j in self.active:
            for row in self.rows[j]:
                for stage in (j,j+1):
                    for col in (*thermal_columns(stage),*self.columns.get(stage,())):
                        matrix.mark(row,col)

    def add_local_derivatives(self,add,j,column,props,changed,step,vapor_flows):
        if not self.active:
            return
        for i,c in enumerate(self.components[:-1]):
            dy = (changed['y'][c]-props[j]['y'][c])/step
            if j in self.rows:
                dyeq = (changed['equilibrium_y'][c]-props[j]['equilibrium_y'][c])/step
                add(self.rows[j][i],column,dy-self.values[j]*dyeq)
            if j-1 in self.rows:
                _,weight = self.incoming(j-1,vapor_flows[j],props[j]['y'])
                add(self.rows[j-1][i],column,-(1-self.values[j-1])*weight*dy)

    def add_flow_derivatives(self,add,j,column,props,vapor_flows):
        if j-1 not in self.rows:
            return
        yin,_ = self.incoming(j-1,vapor_flows[j],props[j]['y'])
        total = vapor_flows[j]+self.feed_flow[j-1]
        for i,c in enumerate(self.components[:-1]):
            add(self.rows[j-1][i],column,-(1-self.values[j-1])*vapor_flows[j]*(props[j]['y'][c]-yin[c])/total)

    def diagnostics(self,props,vapor_flows):
        errors = self.residuals(props,vapor_flows,all_components=True)
        error = max(map(abs,errors),default=0.)
        target = float(self.unit.get_param('mesh_tolerance',2e-6))
        tolerance = max(target,float(self.unit.get_param('acceptable_mesh_residual',50.*target)))
        if not math.isfinite(error) or error > len(self.components)*tolerance:
            raise UnitOperationError(f'{self.unit.unit_id}: post-solve vapor Murphree residual {error:.3e} exceeds tolerance')
        return {'stage_efficiency_model':'murphree_vapor',
                'stage_efficiencies':list(self.values),
                'stage_equilibrium_vapor_compositions':[dict(p.get('equilibrium_y',p['y'])) for p in props],
                'max_murphree_residual':error}
