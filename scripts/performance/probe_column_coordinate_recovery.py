#!/usr/bin/env python3
"""Runtime-only column recovery experiments; production files are never edited.

Uses authoritative MESH residuals in alternative coordinates and projections.
mass_sweep combines stable bottom-inventory coordinates, projected temperatures
and energy flows, and tridiagonal fixed-K material solves with Anderson mixing.
Other modes retain failed diagnostic approaches for reproducible comparison.
Colored finite differences reuse the transformed native pattern where needed.
The exact dense trust-region solvers are for diagnostic-size columns.
Workers run sequentially with one BLAS thread. Every attempt and unit result is
persisted immediately. Unknown model layouts retain the actual native solver.

Example:
  .venv/bin/python scripts/performance/probe_column_coordinate_recovery.py \
      --output /tmp/column-coordinate-new --cases ethanol_water ipa_water \
      --factors 1.02 --initializers cheap_estimate azeotropic --mode mass_sweep
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe_azeotropic_overdistillation import candidate_metadata, json_default, prepare_case
from benchmark_sparse_column_solves import prepare_case as normal_case, stream_record
from benchmark_column_trust_region import newton_direction

NORMAL_CASES = ("benzene_toluene_20","hydrocarbon_pr_42","partial_mass_12","absorption_5","stripping_5")
CASES = ("ethanol_water","ipa_water","nitrile_ternary","ethanol_water_propanol",*NORMAL_CASES)


class FlowCoordinates:
    """A bijective interior coordinate change, with nonnegative-flow boundaries."""
    def __init__(self,unit,model,args,x0):
        import numpy as np
        from scipy.sparse import lil_matrix
        self.unit,self.model,self.args = unit,model,args
        self.native_residual_calls = 0
        self.N,self.comps = args["N"],args["comps"]
        self.nc = len(self.comps)
        self.L_start = self.N*self.nc
        self.V_start = self.L_start+self.N
        self.Q_start = self.V_start+self.N
        self.core = self.Q_start+(2 if "condenser" in args else 0)
        if len(x0)!=self.core:
            raise ValueError("extra stage-efficiency variables are not included in this diagnostic")
        self.component_scales = np.array([args["component_scales"][c] for c in self.comps])
        self.flow_scale = args["flow_scale"]
        self.energy_scale = args["energy_scale"]
        self.duty_scale = self.flow_scale*50000.
        B = lil_matrix((len(x0),len(x0)),dtype=float)
        for j in range(self.N):
            B[j,j] = 1
            flow_columns = list(range(self.N+j*self.nc,self.N+(j+1)*self.nc))
            for ci in range(self.nc-1):
                old_col = self.N+j*(self.nc-1)+ci
                B[old_col,flow_columns[ci]] = 1
                B[old_col,flow_columns[-1]] = 1
            B[self.L_start+j,flow_columns] = 1
            B[self.V_start+j,self.V_start+j] = 1
        for col in range(self.Q_start,len(x0)):
            B[col,col] = 1
        self.pattern = (model["sparsity"].astype(float) @ B.tocsr()).astype(bool).astype(float).tocsr()
        self.groups = unit._color_jacobian_columns(self.pattern)
        self.lower = np.zeros(len(x0))
        self.upper = np.full(len(x0),np.inf)
        self.lower[:self.N],self.upper[:self.N] = -60.,60.
        self.lower[self.Q_start:],self.upper[self.Q_start:] = -np.inf,np.inf
        # Remove constant reference-enthalpy terms with component-balance rows.
        # This triangular row transform preserves the original zero set.
        self.P = lil_matrix((len(x0),len(x0)),dtype=float)
        self.P.setdiag(np.ones(len(x0)))
        self.reference_enthalpies = np.array([unit.thermo.mixture_enthalpy(
            {v:float(v==c) for v in self.comps},298.15,0.,P=args["pressures"][0]) for c in self.comps])
        for j in range(self.N):
            row = j*(self.nc+2)+self.nc
            self.P[row,row] = self.energy_scale/self.duty_scale
            for ci,c in enumerate(self.comps):
                reference = self.reference_enthalpies[ci]
                self.P[row,j*(self.nc+2)+ci] = -reference*self.component_scales[ci]/self.duty_scale
        self.P = self.P.tocsr()
        # Energy rows already contain the component-row supports in native MESH.
        self.pattern = (abs(self.P) @ self.pattern).astype(bool).astype(float).tocsr()
        self.groups = unit._color_jacobian_columns(self.pattern)

    def pack(self,native):
        import numpy as np
        decoded = self.model["decode"](native)
        u = np.empty(len(native))
        u[:self.N] = native[:self.N]
        for j in range(self.N):
            for ci,c in enumerate(self.comps):
                u[self.N+j*self.nc+ci] = decoded["L"][j]*decoded["x"][j][c]/self.component_scales[ci]
        u[self.V_start:self.Q_start] = np.asarray(decoded["V"])/self.flow_scale
        if self.Q_start<len(u):
            u[self.Q_start:] = native[self.Q_start:]*self.energy_scale/self.duty_scale
        return u

    def native(self,u):
        import numpy as np
        z = np.empty(len(u))
        z[:self.N] = u[:self.N]
        for j in range(self.N):
            flows = u[self.N+j*self.nc:self.N+(j+1)*self.nc]*self.component_scales
            total = float(np.sum(flows))
            if total<=0 or np.any(flows<0):
                raise ValueError("nonpositive total liquid flow or negative component flow")
            logs = np.log(np.maximum(flows,1e-300))
            start = self.N+j*(self.nc-1)
            z[start:start+self.nc-1] = logs[:-1]-logs[-1]
            z[self.L_start+j] = math.log(total)
        z[self.V_start:self.Q_start] = np.log(np.maximum(u[self.V_start:self.Q_start]*self.flow_scale,1e-300))
        if self.Q_start<len(u):
            z[self.Q_start:] = u[self.Q_start:]*self.duty_scale/self.energy_scale
        return z

    def residual(self,u):
        return self.P @ self.evaluate_native(self.native(u))

    def evaluate_native(self,z):
        self.native_residual_calls += 1
        return self.model["residual"](z)


class ProjectedCoordinates(FlowCoordinates):
    """Eliminate prescribed flows and all tray total-material balances exactly."""
    def __init__(self,unit,model,args,x0):
        import numpy as np
        from scipy.sparse import lil_matrix
        super().__init__(unit,model,args,x0)
        if "condenser" not in args or args["side_draws"]:
            raise ValueError("projection is for distillation without side draws")
        self.old_size = len(x0)
        self.new_flow_start = self.N*self.nc
        self.new_Q_start = self.new_flow_start+self.N-2
        self.new_size = self.new_Q_start+2
        feeds = np.zeros(self.N)
        for feed in args["feed_specs"]:
            feeds[feed["stage"]] += feed["F"]*sum(feed["z"].get(c,0.) for c in self.comps)
        self.feeds,self.cumulative = feeds,np.cumsum(feeds)
        self.total = float(np.sum(feeds))
        self.RR = args["RR"]
        self.beta = args["condenser_vapor_fraction"]
        self.spec = args["distillate_spec"]
        B = lil_matrix((self.old_size,self.new_size),dtype=float)
        cut_columns = ([0]+list(range(self.N,self.N+self.nc-1))) if self.spec["kind"]=="mass" else []
        for j in range(self.N):
            B[j,j] = 1
            columns = list(range(self.N+j*(self.nc-1),self.N+(j+1)*(self.nc-1)))
            for ci in range(self.nc-1):
                B[self.N+j*(self.nc-1)+ci,columns] = 1
            for col in cut_columns:
                B[self.L_start+j,col] = 1
                B[self.V_start+j,col] = 1
            if 1<=j<=self.N-2:
                B[self.L_start+j,self.new_flow_start+j-1] = 1
            if j>=2:
                B[self.V_start+j,self.new_flow_start+j-2] = 1
        B[self.Q_start,self.new_Q_start] = 1
        B[self.Q_start+1,self.new_Q_start+1] = 1
        self.keep = np.array([j*(self.nc+2)+ci for j in range(self.N)
                             for ci in (*range(self.nc-1),self.nc,self.nc+1)])
        self.pattern = (abs(self.P) @ model["sparsity"].astype(float) @ B.tocsr())[self.keep].astype(bool).astype(float).tocsr()
        self.groups = unit._color_jacobian_columns(self.pattern)
        self.lower,self.upper = np.zeros(self.new_size),np.full(self.new_size,np.inf)
        self.lower[:self.N],self.upper[:self.N] = -60.,60.
        self.upper[self.N:self.new_flow_start] = 1.
        self.lower[self.new_flow_start:self.new_Q_start] = math.exp(-40)/self.flow_scale
        self.lower[self.new_Q_start:] = -np.inf

    def pack(self,z):
        import numpy as np
        decoded = self.model["decode"](z)
        u = np.empty(self.new_size)
        u[:self.N] = z[:self.N]
        for j in range(self.N):
            remaining = 1.
            for ci,c in enumerate(self.comps[:-1]):
                fraction = min(max(decoded["x"][j][c]/max(remaining,1e-300),0.),1.)
                u[self.N+j*(self.nc-1)+ci] = fraction
                remaining *= 1-fraction
        D = decoded["V"][0]
        for j in range(1,self.N-1):
            value = .5*(decoded["L"][j]+decoded["V"][j+1]-abs(self.cumulative[j]-D))
            u[self.new_flow_start+j-1] = max(value,math.exp(-40))/self.flow_scale
        u[self.new_Q_start:] = z[self.Q_start:]*self.energy_scale/self.duty_scale
        return u

    def native(self,u):
        import numpy as np
        z = np.empty(self.old_size)
        z[:self.N] = u[:self.N]
        compositions = []
        for j in range(self.N):
            remaining = 1.
            fractions = []
            for ci in range(self.nc-1):
                value = u[self.N+j*(self.nc-1)+ci]
                fractions.append(remaining*value)
                remaining *= 1-value
            fractions.append(remaining)
            compositions.append(dict(zip(self.comps,fractions)))
            logs = np.log(np.maximum(fractions,1e-300))
            start = self.N+j*(self.nc-1)
            z[start:start+self.nc-1] = logs[:-1]-logs[-1]
        if self.spec["kind"]=="molar":
            D = self.spec["value"]
        else:
            span = self.args["T_max"]-self.args["T_min"]
            T = self.args["T_min"]+span/(1+math.exp(-float(u[0])))
            x = compositions[0]
            y = self.model["stage_properties"](0,T,x)["y"]
            mw = sum(((1-self.beta)*x[c]+self.beta*y[c])*self.unit.thermo.props[c].MW for c in self.comps)
            D = self.spec["value"]/mw
        if not 0<D<self.total:
            raise ValueError("projected product cut leaves no positive bottoms")
        L,V = np.empty(self.N),np.empty(self.N)
        L[0],L[-1] = self.RR*D,self.total-D
        V[0],V[1] = D,(self.RR+1)*D-self.feeds[0]
        if V[1]<0:
            raise ValueError("projected condenser vapor flow is negative")
        for j in range(1,self.N-1):
            excess = u[self.new_flow_start+j-1]*self.flow_scale
            balance = self.cumulative[j]-D
            L[j] = excess+max(balance,0.)
            V[j+1] = excess+max(-balance,0.)
        z[self.L_start:self.V_start] = np.log(np.maximum(L,math.exp(-40)))
        z[self.V_start:self.Q_start] = np.log(np.maximum(V,math.exp(-40)))
        z[self.Q_start:] = u[self.new_Q_start:]*self.duty_scale/self.energy_scale
        return z

    def residual(self,u):
        return (self.P @ self.evaluate_native(self.native(u)))[self.keep]


class InventoryCoordinates(ProjectedCoordinates):
    """Also eliminate bottom composition using exact overall component inventories."""
    def __init__(self,unit,model,args,x0):
        import numpy as np
        from scipy.sparse import lil_matrix
        super().__init__(unit,model,args,x0)
        if args["condenser"]!="total":
            raise ValueError("inventory projection currently requires a total condenser")
        self.inventory = np.array([sum(f["F"]*f["z"].get(c,0.) for f in args["feed_specs"]) for c in self.comps])
        self.mass_weights = np.array([unit.thermo.props[c].MW if self.spec["kind"]=="mass" else 1. for c in self.comps])
        self.capacity = self.inventory*self.mass_weights
        self.cut_amount = self.spec["value"]
        self.new_flow_start = self.N+(self.N-1)*(self.nc-1)
        self.new_Q_start = self.new_flow_start+self.N-2
        self.new_size = self.new_Q_start+2
        B = lil_matrix((self.old_size,self.new_size),dtype=float)
        top_columns = list(range(self.N,self.N+self.nc-1))
        for j in range(self.N):
            B[j,j] = 1
            columns = top_columns if j in (0,self.N-1) else list(range(self.N+j*(self.nc-1),self.N+(j+1)*(self.nc-1)))
            for ci in range(self.nc-1):
                B[self.N+j*(self.nc-1)+ci,columns] = 1
            if self.spec["kind"]=="mass":
                for col in top_columns:
                    B[self.L_start+j,col] = 1
                    B[self.V_start+j,col] = 1
            if 1<=j<=self.N-2:
                B[self.L_start+j,self.new_flow_start+j-1] = 1
            if j>=2:
                B[self.V_start+j,self.new_flow_start+j-2] = 1
        B[self.Q_start,self.new_Q_start] = 1
        B[self.Q_start+1,self.new_Q_start+1] = 1
        self.native_support = B.tocsr()
        self.keep = np.array([j*(self.nc+2)+ci for j in range(self.N)
            for ci in ((*range(self.nc-1),self.nc,self.nc+1) if j<self.N-1 else (self.nc,self.nc+1))])
        self.pattern = (abs(self.P) @ model["sparsity"].astype(float) @ B.tocsr())[self.keep].astype(bool).astype(float).tocsr()
        self.groups = unit._color_jacobian_columns(self.pattern)
        self.lower,self.upper = np.zeros(self.new_size),np.full(self.new_size,np.inf)
        self.lower[:self.N],self.upper[:self.N] = -60.,60.
        self.upper[self.N:self.new_flow_start] = 1.
        self.lower[self.new_flow_start:self.new_Q_start] = math.exp(-40)/self.flow_scale
        self.lower[self.new_Q_start:] = -np.inf

    def pack(self,z):
        import numpy as np
        decoded = self.model["decode"](z)
        u = np.empty(self.new_size)
        u[:self.N] = z[:self.N]
        for j in range(1,self.N-1):
            remainder = 1.
            for ci,c in enumerate(self.comps[:-1]):
                value = min(max(decoded["x"][j][c]/max(remainder,1e-300),0.),1.)
                u[self.N+j*(self.nc-1)+ci] = self.encode_fraction(value)
                remainder *= 1-value
        D = decoded["V"][0]
        for j in range(1,self.N-1):
            value = .5*(decoded["L"][j]+decoded["V"][j+1]-abs(self.cumulative[j]-D))
            u[self.new_flow_start+j-1] = max(value,math.exp(-40))/self.flow_scale
        u[self.new_Q_start:] = z[self.Q_start:]*self.energy_scale/self.duty_scale
        u[self.N:self.N+self.nc-1] = self.encode_products(decoded)
        return u

    def encode_fraction(self,value):
        return value

    def fraction(self,value):
        return value

    def encode_products(self,decoded):
        import numpy as np
        desired = np.array([decoded["V"][0]*decoded["x"][0][c] for c in self.comps])*self.mass_weights
        values = np.empty(self.nc-1)
        remaining = self.cut_amount
        for ci in range(self.nc-1):
            low = max(0.,remaining-float(np.sum(self.capacity[ci+1:])))
            high = min(self.capacity[ci],remaining)
            selected = min(max(desired[ci],low),high)
            values[ci] = (selected-low)/(high-low) if high>low else 0.
            remaining -= selected
        return values

    def product_moles(self,u):
        import numpy as np
        allocations = np.empty(self.nc)
        remaining = self.cut_amount
        for ci in range(self.nc-1):
            low = max(0.,remaining-float(np.sum(self.capacity[ci+1:])))
            high = min(self.capacity[ci],remaining)
            allocations[ci] = low+(high-low)*self.fraction(u[self.N+ci])
            remaining -= allocations[ci]
        allocations[-1] = remaining
        dist_moles = allocations/self.mass_weights
        return dist_moles,self.inventory-dist_moles

    def native(self,u):
        import numpy as np
        z = np.empty(self.old_size)
        z[:self.N] = u[:self.N]
        dist_moles,bottom_moles = self.product_moles(u)
        D = float(np.sum(dist_moles))
        B = float(np.sum(bottom_moles))
        if D<=0 or B<=0:
            raise ValueError("projected inventory leaves an empty product")
        for j in range(self.N):
            if j==0:
                fractions = dist_moles/D
            elif j==self.N-1:
                fractions = bottom_moles/B
            else:
                remaining_fraction = 1.
                fractions = []
                for ci in range(self.nc-1):
                    value = self.fraction(u[self.N+j*(self.nc-1)+ci])
                    fractions.append(remaining_fraction*value)
                    remaining_fraction *= 1-value
                fractions.append(remaining_fraction)
            logs = np.log(np.maximum(fractions,1e-300))
            start = self.N+j*(self.nc-1)
            z[start:start+self.nc-1] = logs[:-1]-logs[-1]
        L,V = np.empty(self.N),np.empty(self.N)
        L[0],L[-1] = self.RR*D,B
        V[0],V[1] = D,(self.RR+1)*D-self.feeds[0]
        if V[1]<0:
            raise ValueError("projected condenser vapor flow is negative")
        for j in range(1,self.N-1):
            excess = u[self.new_flow_start+j-1]*self.flow_scale
            balance = self.cumulative[j]-D
            L[j] = excess+max(balance,0.)
            V[j+1] = excess+max(-balance,0.)
        z[self.L_start:self.V_start] = np.log(np.maximum(L,math.exp(-40)))
        z[self.V_start:self.Q_start] = np.log(np.maximum(V,math.exp(-40)))
        z[self.Q_start:] = u[self.new_Q_start:]*self.duty_scale/self.energy_scale
        return z


class EquilibriumCoordinates(InventoryCoordinates):
    """Project bubble temperatures and unconstrained end duties algebraically."""
    def __init__(self,unit,model,args,x0):
        import numpy as np
        from scipy.sparse import lil_matrix
        super().__init__(unit,model,args,x0)
        self.parent_size = self.new_size
        self.parent_flow_start = self.new_flow_start
        self.parent_Q_start = self.new_Q_start
        reduced_size = self.parent_size-self.N-2
        C = lil_matrix((self.parent_size,reduced_size),dtype=float)
        top = list(range(self.nc-1))
        for j in range(self.N):
            columns = top if j in (0,self.N-1) else list(range(j*(self.nc-1),(j+1)*(self.nc-1)))
            C[j,columns] = 1
        for col in range(self.N,self.parent_Q_start):
            C[col,col-self.N] = 1
        self.keep = np.array([j*(self.nc+2)+ci for j in range(self.N)
            for ci in ((*range(self.nc-1),self.nc) if 0<j<self.N-1 else range(self.nc-1) if j==0 else ())])
        self.pattern = (abs(self.P) @ model["sparsity"].astype(float) @ self.native_support @ C.tocsr())[self.keep].astype(bool).astype(float).tocsr()
        self.groups = unit._color_jacobian_columns(self.pattern)
        self.lower = self.lower[self.N:self.parent_Q_start].copy()
        self.upper = self.upper[self.N:self.parent_Q_start].copy()
        self.temperature_cache = {}
        self.last_temperatures = list(model["decode"](x0)["T"])

    def pack(self,z):
        return super().pack(z)[self.N:self.parent_Q_start]

    def native(self,u):
        import numpy as np
        parent = np.zeros(self.parent_size)
        parent[self.N:self.parent_Q_start] = u
        z = super().native(parent)
        self.project_temperatures(z)
        raw = self.evaluate_native(z)
        energy_rows = [self.nc,(self.N-1)*(self.nc+2)+self.nc]
        z[self.Q_start:] = -raw[energy_rows]
        return z

    def project_temperatures(self,z):
        decoded = self.model["decode"](z)
        Tmin,Tmax = self.args["T_min"],self.args["T_max"]
        for j in range(self.N):
            x = decoded["x"][j]
            key = (j,tuple(x[c] for c in self.comps))
            T = self.temperature_cache.get(key)
            if T is None:
                from scipy.optimize import brentq
                def bubble(value):
                    return self.model["stage_properties"](j,value,x)["bubble"]
                guess = self.last_temperatures[j]
                radius = 10.
                while True:
                    low,high = max(Tmin,guess-radius),min(Tmax,guess+radius)
                    if bubble(low)*bubble(high)<=0:
                        T = brentq(bubble,low,high,xtol=1e-9,rtol=1e-12)
                        break
                    if low==Tmin and high==Tmax:
                        raise ValueError("no projected bubble root inside original temperature bounds")
                    radius *= 2
                self.temperature_cache[key] = T
                self.last_temperatures[j] = T
            fraction = min(max((T-Tmin)/(Tmax-Tmin),1e-14),1-1e-14)
            z[j] = math.log(fraction/(1-fraction))


class InsideOutCoordinates(EquilibriumCoordinates):
    """Solve adjusted interior energy balances exactly for all phase flows."""
    def __init__(self,unit,model,args,x0):
        import numpy as np
        from scipy.sparse import csr_matrix
        super().__init__(unit,model,args,x0)
        self.composition_size = (self.N-1)*(self.nc-1)
        self.lower = self.lower[:self.composition_size]
        self.upper = self.upper[:self.composition_size]
        self.keep = np.array([j*(self.nc+2)+ci for j in range(self.N-1) for ci in range(self.nc-1)])
        # The forward energy recurrence couples composition variables globally.
        self.pattern = csr_matrix(np.ones((self.composition_size,self.composition_size)))
        self.groups = [[col] for col in range(self.composition_size)]

    def pack(self,z):
        return super().pack(z)[:self.composition_size]

    def native(self,u):
        import numpy as np
        parent = np.zeros(self.parent_size)
        parent[self.N:self.N+self.composition_size] = u
        z = InventoryCoordinates.native(self,parent)
        self.project_temperatures(z)
        decoded = self.model["decode"](z)
        properties = [self.model["stage_properties"](j,decoded["T"][j],decoded["x"][j]) for j in range(self.N)]
        hL = np.array([p["hL"]-sum(decoded["x"][j][c]*self.reference_enthalpies[ci] for ci,c in enumerate(self.comps))
                       for j,p in enumerate(properties)])
        hV = np.array([p["hV"]-sum(p["y"][c]*self.reference_enthalpies[ci] for ci,c in enumerate(self.comps)) for p in properties])
        feeds = np.zeros(self.N)
        for feed in self.args["feed_specs"]:
            reference = sum(feed["z"].get(c,0.)*self.reference_enthalpies[ci] for ci,c in enumerate(self.comps))
            feeds[feed["stage"]] += feed["F"]*(feed["H"]-reference)
        L,V = np.array(decoded["L"]),np.array(decoded["V"])
        D = V[0]
        for j in range(1,self.N-1):
            offset = self.cumulative[j]-D
            constant = L[j-1]*hL[j-1]-V[j]*hV[j]+feeds[j]+max(-offset,0.)*hV[j+1]-max(offset,0.)*hL[j]
            coefficient = hV[j+1]-hL[j]
            if abs(coefficient)<1e-6:
                raise ValueError("vanishing projected latent-heat coefficient")
            excess = -constant/coefficient
            if excess<=0:
                raise ValueError("negative phase flow from projected energy recurrence")
            L[j] = excess+max(offset,0.)
            V[j+1] = excess+max(-offset,0.)
        z[self.L_start:self.V_start] = np.log(L)
        z[self.V_start:self.Q_start] = np.log(V)
        raw = self.evaluate_native(z)
        z[self.Q_start:] = -raw[[self.nc,(self.N-1)*(self.nc+2)+self.nc]]
        return z


class StableLogCoordinates(InsideOutCoordinates):
    """Store small bottom allocations directly, avoiding one-minus-recovery loss."""
    log_chart = True

    def __init__(self,*args):
        import numpy as np
        super().__init__(*args)
        self.bottom_cut = float(np.sum(self.capacity))-self.cut_amount
        self.lower = np.full(self.composition_size,-700.)
        self.upper = np.zeros(self.composition_size)

    def encode_fraction(self,value):
        return math.log(max(value,math.exp(-700)))

    def fraction(self,value):
        return math.exp(float(value))

    def encode_bottoms(self,desired):
        import numpy as np
        values = np.empty(self.nc-1)
        remaining = self.bottom_cut
        for ci in range(self.nc-1):
            low = max(0.,remaining-float(np.sum(self.capacity[ci+1:])))
            high = min(self.capacity[ci],remaining)
            selected = min(max(desired[ci],low),high)
            ratio = (selected-low)/(high-low) if high>low else 0.
            values[ci] = self.encode_fraction(ratio)
            remaining -= selected
        return values

    def encode_products(self,decoded):
        import numpy as np
        desired = np.array([decoded["L"][-1]*decoded["x"][-1][c] for c in self.comps])*self.mass_weights
        return self.encode_bottoms(desired)

    def product_moles(self,u):
        import numpy as np
        allocations = np.empty(self.nc)
        remaining = self.bottom_cut
        for ci in range(self.nc-1):
            low = max(0.,remaining-float(np.sum(self.capacity[ci+1:])))
            high = min(self.capacity[ci],remaining)
            allocations[ci] = low+(high-low)*self.fraction(u[self.N+ci])
            remaining -= allocations[ci]
        allocations[-1] = remaining
        bottom = allocations/self.mass_weights
        return self.inventory-bottom,bottom


def binary_region(coordinates):
    """Restrict homogeneous constant-pressure binaries to the feed's azeotropic region."""
    import numpy as np
    if getattr(coordinates,"log_chart",False) and not getattr(coordinates,"_physical_region",False):
        coordinates._physical_region = True
        coordinates.lower = np.zeros(coordinates.composition_size)
        coordinates.upper = np.ones(coordinates.composition_size)
        info = binary_region(coordinates)
        lower,upper = coordinates.lower.copy(),coordinates.upper.copy()
        lower[0],upper[0] = 1-upper[0],1-lower[0]
        coordinates.lower = np.log(np.maximum(lower,math.exp(-700)))
        coordinates.upper = np.log(np.maximum(upper,math.exp(-700)))
        coordinates._physical_region = False
        info["chart"] = "log bottom allocation and log internal fractions"
        return info
    if not isinstance(coordinates,InventoryCoordinates) or coordinates.nc!=2:
        return {"applied":False,"reason":"not a binary total-condenser inventory model"}
    a = coordinates.args
    if np.ptp(a["pressures"])>1e-12:
        return {"applied":False,"reason":"pressure varies across stages"}
    unit = coordinates.unit
    candidates = unit._vle_azeotrope_candidates(coordinates.comps,a["pressures"][0])
    points = []
    for candidate in candidates:
        splitter = getattr(unit.thermo,"liquid_liquid_equilibrium",None)
        if splitter and splitter(candidate["composition"],candidate["T"])[0]:
            continue
        points.append(candidate["composition"][coordinates.comps[0]])
    feed = coordinates.inventory[0]/coordinates.total
    if not points or any(abs(feed-point)<1e-6 for point in points):
        return {"applied":False,"reason":"no separated homogeneous azeotropic region"}
    lo = max([0.]+[point for point in points if point<feed])
    hi = min([1.]+[point for point in points if point>feed])
    lo,hi = max(0.,lo-1e-7),min(1.,hi+1e-7)
    offset = 0 if isinstance(coordinates,EquilibriumCoordinates) else coordinates.N
    for j in range(1,coordinates.N-1):
        coordinates.lower[offset+j] = lo
        coordinates.upper[offset+j] = hi
    C = coordinates.cut_amount
    low = max(0.,C-coordinates.capacity[1])
    high = min(coordinates.capacity[0],C)
    g0,g1 = coordinates.mass_weights
    def top_threshold(x):
        amount = C*x*g0/(x*g0+(1-x)*g1)
        return (amount-low)/(high-low)
    def bottom_threshold(x):
        amount = g0*(g1*coordinates.inventory[0]-x*g1*coordinates.total+x*C)/(g1*(1-x)+x*g0)
        return (amount-low)/(high-low)
    coordinates.lower[offset] = max(0.,top_threshold(lo),bottom_threshold(hi))
    coordinates.upper[offset] = min(1.,top_threshold(hi),bottom_threshold(lo))
    if coordinates.lower[offset]>=coordinates.upper[offset]:
        raise ValueError("degenerate endpoint interval under binary region projection")
    return {"applied":True,"component":coordinates.comps[0],"feed_fraction":feed,
            "interval":[lo,hi],"endpoint_interval":[coordinates.lower[offset],coordinates.upper[offset]]}


def mass_sweep(unit,model,args,x0,options,budget,entry,plain=False):
    """Fixed-K tridiagonal component sweeps with constrained Anderson mixing."""
    import numpy as np
    from thermodynamics_models.common import ThermodynamicsError
    coordinates = StableLogCoordinates(unit,model,args,x0)
    entry["region_projection"] = binary_region(coordinates)
    u = np.clip(coordinates.pack(x0),coordinates.lower,coordinates.upper)
    history,merits,trace = [],[],[]
    entry["sweep_trace"] = trace
    best = None
    message = "sweep budget exhausted"
    iterations,linear_solves = 0,0

    def evaluate(values):
        z = coordinates.native(values)
        f = coordinates.evaluate_native(z)
        return z,f,.5*float(f@f)

    for iteration in range(budget):
        z,f,merit = evaluate(u)
        norm = float(np.linalg.norm(f,ord=np.inf))
        if best is None or norm<best[0]:
            best = (norm,z.copy(),f.copy())
        trace.append({"iteration":iteration,"residual":norm,"best_residual":best[0],"merit":merit})
        if norm<options["mesh_tolerance"]:
            message = "converged"
            break
        if iteration%25==0:
            print("SWEEP",unit.unit_id,iteration,"residual",norm,"best",best[0],flush=True)
        g = material_update(coordinates,u,entry,z)
        linear_solves += coordinates.nc
        difference = g-u
        history.append((g.copy(),difference.copy()))
        history = history[-6:]
        proposals = []
        if len(history)>1 and not plain:
            G = np.column_stack([history[i+1][0]-history[i][0] for i in range(len(history)-1)])
            R = np.column_stack([history[i+1][1]-history[i][1] for i in range(len(history)-1)])
            weights = np.linalg.lstsq(R,difference,rcond=1e-8)[0]
            proposals.append(g-G@weights)
        proposals += [u+mix*difference for mix in ((.5,.25,.1,.05,.01) if plain else (1.,.5,.25,.1,.05,.01))]
        trials = []
        reference = max(merits[-8:]+[merit])
        for proposed in proposals:
            proposed = np.clip(proposed,coordinates.lower,coordinates.upper)
            try:
                zz,ff,trial_merit = evaluate(proposed)
            except (ValueError,ThermodynamicsError,OverflowError):
                continue
            if np.all(np.isfinite(ff)):
                trials.append((trial_merit,proposed))
                trial_norm = float(np.linalg.norm(ff,ord=np.inf))
                if best is None or trial_norm<best[0]:
                    best = (trial_norm,zz.copy(),ff.copy())
                if plain or trial_merit<reference:
                    break
        if not trials:
            message = "no feasible sweep update"
            break
        _,u = min(trials,key=lambda item:item[0])
        merits.append(merit)
        iterations = iteration+1
    norm,z,f = best
    entry.update(coordinate_strategy="StableLogCoordinates",coordinate_count=len(u),native_coordinate_count=len(z),
                 linear_component_solves=linear_solves,native_vector=z.tolist(),
                 stage_liquid_compositions=model["decode"](z)["x"])
    return {"x":z,"success":bool(norm<options["mesh_tolerance"]),"residual_norm":norm,
            "iterations":iterations,"jacobian_evaluations":0,"function_evaluations":coordinates.native_residual_calls,
            "jacobian_method":"fixed_K_component_sweep","message":message}


def material_update(coordinates,u,entry,z=None):
    """One globally balanced fixed-K update, shared by sweeps and preconditioned roots."""
    import numpy as np
    from scipy.linalg import solve_banded
    model,args = coordinates.model,coordinates.args
    if z is None:
        z = coordinates.native(u)
    decoded = model["decode"](z)
    props = [model["stage_properties"](j,decoded["T"][j],decoded["x"][j]) for j in range(coordinates.N)]
    L,V = np.array(decoded["L"]),np.array(decoded["V"])
    rhs = np.zeros((coordinates.N,coordinates.nc))
    for feed in args["feed_specs"]:
        for ci,c in enumerate(coordinates.comps):
            rhs[feed["stage"],ci] += feed["F"]*feed["z"].get(c,0.)
    amounts = np.empty_like(rhs)
    for ci,c in enumerate(coordinates.comps):
        K = np.array([p["K"][c]/sum(p["K"][v]*decoded["x"][j][v] for v in coordinates.comps) for j,p in enumerate(props)])
        matrix = np.zeros((3,coordinates.N))
        matrix[1] = L+V*K
        matrix[1,0] = L[0]+V[0]
        matrix[0,1:] = -V[1:]*K[1:]
        matrix[2,:-1] = -L[:-1]
        amounts[:,ci] = solve_banded((1,1),matrix,rhs[:,ci])
    entry["linear_component_solves"] = entry.get("linear_component_solves",0)+coordinates.nc
    if np.any(amounts<0) or not np.all(np.isfinite(amounts)):
        raise ValueError("nonpositive or nonfinite fixed-K material solution")
    balanced = V[0]*amounts[0]+L[-1]*amounts[-1]
    entry["max_linear_inventory_error"] = max(entry.get("max_linear_inventory_error",0.),float(np.max(np.abs(balanced-coordinates.inventory))))
    compositions = amounts/np.sum(amounts,axis=1)[:,None]
    g = np.empty_like(u)
    g[:coordinates.nc-1] = coordinates.encode_bottoms(L[-1]*amounts[-1]*coordinates.mass_weights)
    for j in range(1,coordinates.N-1):
        remaining = 1.
        for ci in range(coordinates.nc-1):
            fraction = min(max(compositions[j,ci]/max(remaining,1e-300),0.),1.)
            g[j*(coordinates.nc-1)+ci] = coordinates.encode_fraction(fraction)
            remaining *= 1-fraction
    return np.clip(g,coordinates.lower,coordinates.upper)


def material_root(unit,model,args,x0,options,budget,entry,hybrid=False):
    """Newton/trust-region on relative fixed-K corrections, with full MESH acceptance."""
    import numpy as np
    from scipy.optimize import least_squares,root
    from thermodynamics_models.common import ThermodynamicsError
    coordinates = StableLogCoordinates(unit,model,args,x0)
    entry["region_projection"] = binary_region(coordinates)
    u = np.clip(coordinates.pack(x0),coordinates.lower,coordinates.upper)
    best = [math.inf,None]
    calls = 0

    def residual(values):
        nonlocal calls
        calls += 1
        if hybrid and calls%100==0:
            print("MATERIAL HYBR",unit.unit_id,calls,"best residual",best[0],flush=True)
        try:
            feasible = np.clip(values,coordinates.lower,coordinates.upper)
            z = coordinates.native(feasible)
            raw = coordinates.evaluate_native(z)
            norm = float(np.linalg.norm(raw,ord=np.inf))
            if norm<best[0]:
                best[:] = [norm,z.copy()]
            if hybrid and norm<options["mesh_tolerance"]:
                raise StopIteration
            return material_update(coordinates,feasible,entry,z)-values
        except (ValueError,ThermodynamicsError,OverflowError):
            entry["invalid_trials"] = entry.get("invalid_trials",0)+1
            return np.full_like(values,np.nan)

    iteration = 0
    def callback(intermediate_result):
        nonlocal iteration
        iteration += 1
        if iteration%10==0:
            print("MATERIAL ROOT",unit.unit_id,iteration,"best residual",best[0],flush=True)
        if best[0]<options["mesh_tolerance"] or iteration>=budget:
            raise StopIteration

    if hybrid:
        try:
            result = root(residual,u,method="hybr",options={"factor":.1,"xtol":1e-9,
                "eps":1e-10,"maxfev":budget*(len(u)+1)})
            message = str(result.message)
        except StopIteration:
            message = "original MESH tolerance reached"
        jacobians = None
    else:
        result = least_squares(residual,u,jac="2-point",method="trf",tr_solver="exact",x_scale=1.,
                               bounds=(coordinates.lower,coordinates.upper),ftol=None,gtol=1e-12,xtol=1e-12,
                               max_nfev=budget*16,callback=callback)
        message,jacobians = str(result.message),result.njev
    norm,z = best
    if z is None:
        raise ValueError("material-root formulation has no valid initial evaluation")
    entry.update(coordinate_strategy="StableLogCoordinates",coordinate_count=len(u),native_coordinate_count=len(z),
                 native_vector=z.tolist(),stage_liquid_compositions=model["decode"](z)["x"],material_map_evaluations=calls)
    return {"x":z,"success":bool(norm<options["mesh_tolerance"]),"residual_norm":norm,
            "iterations":iteration if not hybrid else calls,"jacobian_evaluations":jacobians,
            "function_evaluations":coordinates.native_residual_calls,"jacobian_method":"fixed_K_map_fd",
            "message":message}


def coordinate_newton(unit,residual,sparsity,x0,options,jacobian,step_event,entry):
    """Retain useful Newton components when one log-coordinate step is enormous."""
    import numpy as np
    from thermodynamics_models.common import ThermodynamicsError
    x = np.array(x0,dtype=float)
    f = residual(x)
    if not np.all(np.isfinite(f)):
        raise ValueError("nonfinite initial residual")
    functions,jacobians,iterations = 1,0,0
    groups,label = None,"colored_finite_difference"
    message = "iteration budget exhausted"
    stall_best,stall_count = math.inf,0
    for iteration in range(options["max_iterations"]):
        current_norm = float(np.linalg.norm(f,ord=np.inf))
        if current_norm<options["mesh_tolerance"]:
            message = "converged"
            break
        if options.get("stall_iterations",0):
            progress = max(options.get("stall_relative_tolerance",1e-4)*stall_best,1e-12)
            if not math.isfinite(stall_best) or current_norm<stall_best-progress:
                stall_count = 0
            else:
                stall_count += 1
            stall_best = min(stall_best,current_norm)
            if stall_count>=options["stall_iterations"]:
                message = "residual stalled"
                break
        if jacobians>=options["max_jacobian_evaluations"]:
            break
        result = jacobian(x,f,options["finite_difference_rel_step"]) if jacobian else None
        if result is None:
            if groups is None:
                groups = unit._color_jacobian_columns(sparsity)
            J,evaluations = unit._finite_difference_jacobian(residual,x,f,sparsity,groups,options["finite_difference_rel_step"])
        else:
            J,evaluations,*labels = result
            label = labels[0] if labels else "semi_analytic_flow"
        functions += evaluations
        jacobians += 1
        direction = newton_direction(J,f)
        if direction is None:
            message = "Newton system failed"
            break
        if step_event:
            try:
                step_event(x,f,direction)
            except Exception as error:
                progress = getattr(error,"add_solver_progress",None)
                if callable(progress):
                    progress(iterations=iterations,function_evaluations=functions,jacobian_evaluations=jacobians)
                raise
        p = np.clip(direction,-8.,8.)
        gradient = np.asarray(J.T@f).ravel()
        slope = float(gradient@p)
        if slope>=0:
            diagonal = np.asarray(J.power(2).sum(axis=0)).ravel()
            p = np.clip(-gradient/np.maximum(diagonal,1e-12),-8.,8.)
            slope = float(gradient@p)
            entry["gradient_fallbacks"] = entry.get("gradient_fallbacks",0)+1
        merit = .5*float(f@f)
        best = None
        for attempt in range(options["line_search_steps"]):
            alpha = .5**attempt
            trial_x = x+alpha*p
            functions += 1
            try:
                trial_f = residual(trial_x)
            except ThermodynamicsError:
                continue
            if not np.all(np.isfinite(trial_f)):
                continue
            trial_merit = .5*float(trial_f@trial_f)
            if trial_merit<merit and (best is None or trial_merit<best[0]):
                best = (trial_merit,trial_x,trial_f)
            if trial_merit<=merit+1e-4*alpha*slope:
                break
        if best is None:
            message = "coordinatewise step could not reduce merit"
            break
        _,x,f = best
        iterations = iteration+1
    norm = float(np.linalg.norm(f,ord=np.inf))
    acceptable = max(options["mesh_tolerance"],options.get("acceptable_mesh_residual",options["mesh_tolerance"]))
    return {"x":x,"success":bool(norm<acceptable),"residual_norm":norm,
            "iterations":iterations,"jacobian_evaluations":jacobians,"function_evaluations":functions,
            "jacobian_method":label,"message":message}


def box_trust(unit,residual,sparsity,x0,options,jacobian,step_event,entry,model_info=None):
    """Solve the bounded Gauss-Newton subproblem instead of clipping a Newton step."""
    import numpy as np
    from scipy.optimize import lsq_linear
    from thermodynamics_models.common import ThermodynamicsError
    x = np.array(x0,dtype=float)
    f = residual(x)
    functions,jacobians,iterations = 1,0,0
    groups,label = None,"colored_finite_difference"
    P = None
    if model_info:
        _,model,args = model_info
        P = FlowCoordinates(unit,model,args,x0).P
    transform = lambda value: P@value if P is not None else value
    radius = 1.
    message = "Jacobian budget exhausted"
    entry["trust_trace"] = []
    for iteration in range(min(options["max_iterations"],options["max_jacobian_evaluations"])):
        norm = float(np.linalg.norm(f,ord=np.inf))
        if norm<options["mesh_tolerance"]:
            message = "converged"
            break
        result = jacobian(x,f,options["finite_difference_rel_step"]) if jacobian else None
        if result is None:
            if groups is None:
                groups = unit._color_jacobian_columns(sparsity)
            J,evals = unit._finite_difference_jacobian(residual,x,f,sparsity,groups,options["finite_difference_rel_step"])
        else:
            J,evals,*labels = result
            label = labels[0] if labels else "semi_analytic_flow"
        functions += evals
        jacobians += 1
        if step_event:
            try:
                step_event(x,f,newton_direction(J,f))
            except Exception as error:
                progress = getattr(error,"add_solver_progress",None)
                if callable(progress):
                    progress(iterations=iterations,function_evaluations=functions,jacobian_evaluations=jacobians)
                raise
        A = (P@J if P is not None else J).toarray()
        b = np.asarray(transform(f))
        merit = .5*float(b@b)
        accepted = False
        for trial in range(options["line_search_steps"]):
            answer = lsq_linear(A,-b,bounds=(-radius,radius),method="bvls",tol=1e-11,max_iter=200)
            p = answer.x
            Ap = A@p
            predicted = -float(b@Ap)-.5*float(Ap@Ap)
            if predicted<=0 or not math.isfinite(predicted):
                message = "nonpositive bounded-model reduction"
                break
            functions += 1
            try:
                ff = residual(x+p)
            except ThermodynamicsError:
                ff = np.full_like(f,np.nan)
            bb = np.asarray(transform(ff))
            ratio = (merit-.5*float(bb@bb))/predicted if np.all(np.isfinite(ff)) else -math.inf
            step = float(np.max(np.abs(p)))
            entry["trust_trace"].append({"iteration":iteration,"radius":radius,
                                         "ratio":ratio if math.isfinite(ratio) else None,"native_residual":norm})
            if ratio<.25:
                radius = max(.25*step,1e-12)
            elif ratio>.75 and step>.95*radius:
                radius = min(2*radius,8.)
            if ratio>.1:
                x,f = x+p,ff
                iterations = iteration+1
                accepted = True
                break
            if radius<=1e-12:
                message = "box radius underflow"
                break
        if not accepted:
            if message=="Jacobian budget exhausted":
                message = "box trust region could not reduce residual"
            break
    norm = float(np.linalg.norm(f,ord=np.inf))
    entry["native_vector"] = x.tolist()
    if model_info:
        entry["stage_liquid_compositions"] = model_info[1]["decode"](x)["x"]
    return {"x":x,"success":bool(norm<options["mesh_tolerance"]),"residual_norm":norm,
            "iterations":iterations,"jacobian_evaluations":jacobians,"function_evaluations":functions,
            "jacobian_method":label,"message":message}


def recovery_portfolio(unit,residual,sparsity,x0,options,jacobian,step_event,entry,info,native,budget):
    """Cold-start staged recovery, without any prerecorded solution profiles."""
    import numpy as np
    from diagnose_binary_azeotropic_overdraw import dense_solve,profile_seed
    from thermodynamics_models.common import ThermodynamicsError
    _,model,args = info if info else (None,None,None)
    eligible = (args and args.get("condenser")=="total" and not args["side_draws"]
                and len(args["comps"])==2 and len(x0)==args["N"]*4+2
                and not model["condenser_boundary"].options and step_event is None)
    if not eligible:
        entry["recovery_scope"] = "native delegate for unsupported layout or active-set events"
        return native(unit,residual,sparsity,x0,options,jacobian=jacobian,step_event=step_event)
    stages = []
    entry["recovery_stages"] = stages
    best = None
    def run(name,function):
        nonlocal best
        stage = {"name":name}
        stages.append(stage)
        begin = time.perf_counter()
        try:
            answer = function(stage)
            norm = float(np.linalg.norm(residual(answer["x"]),ord=np.inf))
            answer["residual_norm"] = norm
            answer["success"] = norm<options["mesh_tolerance"]
            stage.update({k:v for k,v in answer.items() if k!="x"})
            if best is None or norm<best["residual_norm"]:
                best = answer
            return answer["success"]
        except (ValueError,ThermodynamicsError,RuntimeError,OverflowError) as error:
            stage["error"] = f"{type(error).__name__}: {error}"
            return False
        finally:
            stage["seconds"] = time.perf_counter()-begin
            print("RECOVERY",unit.unit_id,name,stage.get("success"),stage.get("residual_norm"),stage["seconds"],flush=True)

    completed = run("native",lambda stage:native(unit,residual,sparsity,x0,options,jacobian=jacobian,step_event=step_event))
    if not completed:
        def make_seed(initializer,logarithmic=False):
            seed_unit = type(unit)(unit.unit_id,unit.thermo,dict(unit.params,initializer=initializer))
            q = seed_unit._feed_thermal_condition(args["inlet"],args["feed_z"],args["pressures"][args["feed_index"]],args["T_min"],args["T_max"])
            initial = seed_unit._initial_guess(args["inlet"],args["comps"],args["feed_z"],args["N"],args["feed_index"]+1,
                args["RR"],q,args["pressures"],args["condenser"],args["condenser_vapor_fraction"],args["distillate_spec"],
                args["side_draws"],args["T_min"],args["T_max"])
            if logarithmic:
                return profile_seed(seed_unit,{"model_args":args,"initial":initial},False)
            initial["T"][0] = model["condenser_boundary"].seed_temperature(initial["x"][0],initial["T"][0])
            return seed_unit._pack_variables(initial["T"],initial["x"],initial["L"],initial["V"],
                initial["Q_cond"],initial["Q_reb"],args["comps"],args["T_min"],args["T_max"],args["energy_scale"])
        completed = run("balanced_sweep",lambda stage:mass_sweep(unit,model,args,make_seed("cheap_estimate"),options,min(100,budget),stage))
        if not completed:
            problem = {"model":model,"model_args":args,"residual":residual,"jacobian":jacobian}
            dense_answer = []
            def dense_stage(stage):
                seed = make_seed("azeotropic",True)
                answer = dense_solve(problem,seed,True,max_jacobians=min(150,budget))
                answer.update(iterations=answer["jacobian_evaluations"],jacobian_method="dense_native_logit_fd")
                dense_answer.append(answer)
                return answer
            completed = run("log_profile_dense",dense_stage)
            if not completed and dense_answer:
                seed = dense_answer[0]["x"].copy()
                N = args["N"]
                fractions = np.clip(1/(1+np.exp(-seed[N:2*N])),1e-8,1-1e-8)
                seed[N:2*N] = np.log(fractions)-np.log1p(-fractions)
                completed = run("native_refinement",lambda stage:native(unit,residual,sparsity,seed,options,jacobian=jacobian,step_event=step_event))
    if best is None:
        raise ValueError("all recovery stages failed before producing an iterate")
    answer = dict(best)
    answer["iterations"] = sum(s.get("iterations",0) for s in stages)
    answer["jacobian_evaluations"] = sum(s.get("jacobian_evaluations",0) or 0 for s in stages)
    answer["function_evaluations"] = sum(s.get("function_evaluations",0) for s in stages)
    entry["native_vector"] = answer["x"].tolist()
    return answer


def flow_solve(unit,model,args,x0,options,budget,entry,mode):
    import numpy as np
    from scipy.optimize import least_squares
    from thermodynamics_models.common import ThermodynamicsError
    if mode in ("stable_log","stable_log_region") and args.get("condenser")=="total" and not model["condenser_boundary"].options:
        coordinates = StableLogCoordinates(unit,model,args,x0)
    elif mode in ("inside_out","inside_out_region") and args.get("condenser")=="total" and not model["condenser_boundary"].options:
        coordinates = InsideOutCoordinates(unit,model,args,x0)
    elif mode in ("equilibrium","equilibrium_region") and args.get("condenser")=="total" and not model["condenser_boundary"].options:
        coordinates = EquilibriumCoordinates(unit,model,args,x0)
    elif mode in ("inventory","equilibrium","region","equilibrium_region","inside_out","inside_out_region","stable_log","stable_log_region") and args.get("condenser")=="total":
        coordinates = InventoryCoordinates(unit,model,args,x0)
    elif mode in ("projection","inventory","equilibrium","region","equilibrium_region","inside_out","inside_out_region","stable_log","stable_log_region"):
        coordinates = ProjectedCoordinates(unit,model,args,x0)
    else:
        coordinates = FlowCoordinates(unit,model,args,x0)
    entry["coordinate_strategy"] = type(coordinates).__name__
    if mode in ("region","equilibrium_region","inside_out_region","stable_log_region"):
        entry["region_projection"] = binary_region(coordinates)
    u0 = coordinates.pack(x0)
    if u0.shape!=coordinates.lower.shape or not np.all(np.isfinite(u0)):
        raise ValueError("invalid coordinate layout or nonfinite initial projection")
    u0 = np.maximum(np.minimum(u0,np.nextafter(coordinates.upper,coordinates.lower)),np.nextafter(coordinates.lower,coordinates.upper))
    count,functions = 0,0
    cache = [None,None]
    last = [u0.copy(),coordinates.residual(u0)]
    functions = 1

    def fun(u):
        nonlocal functions
        functions += 1
        try:
            f = coordinates.residual(u)
        except (ThermodynamicsError,ValueError,OverflowError):
            entry["invalid_trials"] = entry.get("invalid_trials",0)+1
            f = np.full_like(last[1],np.nan)
        cache[:] = [u.copy(),f]
        return f

    def jac(u):
        nonlocal count,functions
        f = cache[1] if cache[0] is not None and np.array_equal(u,cache[0]) else fun(u)
        count += 1
        # Reflect finite-difference coordinates near upper bounds, reusing the
        # authoritative colored-FD builder with inward feasible perturbations.
        h = 1e-6*np.maximum(np.abs(u),1.)
        signs = np.where(u+h>coordinates.upper,-1.,1.)
        def reflected(v):
            return fun(signs*v)
        J,evaluations = unit._finite_difference_jacobian(reflected,signs*u,f,coordinates.pattern,coordinates.groups,1e-6)
        return J.multiply(signs).toarray()

    def callback(intermediate_result):
        nonlocal functions
        last[:] = [intermediate_result.x.copy(),intermediate_result.fun.copy()]
        native_norm = float(np.linalg.norm(coordinates.evaluate_native(coordinates.native(last[0])),ord=np.inf))
        functions += 1
        if native_norm<options["mesh_tolerance"] or count>=budget:
            raise StopIteration

    result = least_squares(fun,u0,jac=jac,method="trf",tr_solver="exact",x_scale="jac",
                           bounds=(coordinates.lower,coordinates.upper),ftol=None,gtol=1e-13,xtol=1e-13,
                           max_nfev=budget*16,callback=callback)
    z = coordinates.native(result.x)
    norm = float(np.linalg.norm(coordinates.evaluate_native(z),ord=np.inf))
    functions += 1
    entry.update(transformed_merit=.5*float(result.fun@result.fun),coordinate_count=len(u0),native_coordinate_count=len(z),
                 native_vector=z.tolist(),stage_liquid_compositions=model["decode"](z)["x"],
                 fd_groups=len(coordinates.groups))
    return {"x":z,"success":bool(norm<options["mesh_tolerance"]),"residual_norm":norm,
            "iterations":result.njev,"jacobian_evaluations":result.njev,
            "function_evaluations":coordinates.native_residual_calls,"jacobian_method":"colored_component_flow_fd",
            "message":str(result.message)}


def validate_output(unit,inlets,answer,phase_checks=True):
    """Independent conservation and sampled phase checks after native validation."""
    total = sum(s.F for s in inlets.values())
    comps = set().union(*(s.composition for s in inlets.values()))
    checks = {"component_balance_error":max(abs(sum(s.F*s.composition.get(c,0.) for s in inlets.values())-
        sum(s.F*s.composition.get(c,0.) for s in answer.outlet_streams.values()))/max(total,1.) for c in comps)}
    p = answer.performance
    compositions = p.get("stage_liquid_compositions",[])
    temperatures = p.get("stage_temperatures_C",[])
    if phase_checks and compositions and temperatures:
        checker = getattr(unit.thermo,"liquid_spinodal_stability",None)
        if checker:
            states = [checker(T+273.15,x) for T,x in zip(temperatures,compositions)]
            checks["stage_local_stability"] = states
            checks["locally_unstable_stages"] = [j+1 for j,s in enumerate(states) if not s["locally_stable"]]
        splitter = getattr(unit.thermo,"liquid_liquid_equilibrium",None)
        if splitter:
            selected = sorted({0,len(compositions)-1,int(unit.get_param("feed_stage",len(compositions)//2))-1})
            checks["phase_audit"] = []
            for j in selected:
                split,*_ = splitter(compositions[j],temperatures[j]+273.15)
                checks["phase_audit"].append({"stage":j+1,"has_lle":bool(split)})
    return checks


class RuntimePatch:
    def __init__(self,budget,attempts,mode="component_flows"):
        self.budget,self.attempts,self.mode = budget,attempts,mode
        self.models = {}
        self.patches = []

    def __enter__(self):
        from equilibrium_stage_column import EquilibriumStageColumnMixin
        from unit_operations_separation import RigorousAbsorber
        for cls,name in ((EquilibriumStageColumnMixin,"_build_mesh_model"),(RigorousAbsorber,"_build_absorber_mesh_model")):
            original = getattr(cls,name)
            def factory(function):
                def wrapped(unit,*args,**kwargs):
                    values = dict(inspect.signature(function).bind(unit,*args,**kwargs).arguments)
                    model = function(unit,*args,**kwargs)
                    self.models[model["residual"]] = (unit,model,values)
                    return model
                return wrapped
            self.patches.append(patch.object(cls,name,factory(original)))
        native = EquilibriumStageColumnMixin._sparse_newton_solve

        def solver(unit,residual,sparsity,x0,options,jacobian=None,step_event=None):
            begin = time.perf_counter()
            entry = {"unit_id":unit.unit_id,"options":dict(options)}
            self.attempts.append(entry)
            previous = getattr(unit,"_quality_solver_aux_context_active",False)
            unit._quality_solver_aux_context_active = True
            quality = getattr(unit.thermo,"quality_context",None)
            try:
                with quality(phase="solver_iteration",affects_result=False) if quality else nullcontext():
                    info = self.models.get(residual)
                    supported = info is not None and step_event is None
                    if supported:
                        _,_,values = info
                        core = values["N"]*(len(values["comps"])+2)+(2 if "condenser" in values else 0)
                        supported = len(x0)==core
                        if self.mode in ("projection","inventory","equilibrium","region","equilibrium_region","inside_out","inside_out_region","stable_log","stable_log_region","mass_sweep","mass_sweep_plain","material_root","material_hybr"):
                            supported = supported and "condenser" in values and not values["side_draws"]
                        if self.mode in ("mass_sweep","mass_sweep_plain","material_root","material_hybr"):
                            supported = supported and values.get("condenser")=="total" and not info[1]["condenser_boundary"].options
                    if self.mode=="native":
                        answer = native(unit,residual,sparsity,x0,options,jacobian=jacobian,step_event=step_event)
                        entry["algorithm"] = "native"
                    elif self.mode=="portfolio":
                        answer = recovery_portfolio(unit,residual,sparsity,x0,options,jacobian,step_event,entry,info,native,self.budget)
                        entry["algorithm"] = "portfolio"
                    elif self.mode=="coordinate_cap":
                        answer = coordinate_newton(unit,residual,sparsity,x0,options,jacobian,step_event,entry)
                        entry["algorithm"] = "coordinate_cap"
                    elif self.mode=="box_trust":
                        answer = box_trust(unit,residual,sparsity,x0,options,jacobian,step_event,entry,info if supported else None)
                        entry["algorithm"] = "box_trust"
                    elif supported:
                        _,model,values = self.models[residual]
                        if self.mode in ("mass_sweep","mass_sweep_plain"):
                            answer = mass_sweep(unit,model,values,x0,options,self.budget,entry,plain=self.mode=="mass_sweep_plain")
                        elif self.mode in ("material_root","material_hybr"):
                            answer = material_root(unit,model,values,x0,options,self.budget,entry,hybrid=self.mode=="material_hybr")
                        else:
                            answer = flow_solve(unit,model,values,x0,options,self.budget,entry,self.mode)
                        entry["algorithm"] = self.mode
                    else:
                        answer = native(unit,residual,sparsity,x0,options,jacobian=jacobian,step_event=step_event)
                        entry["algorithm"] = "native"
                    entry.update({k:v for k,v in answer.items() if k!="x"})
                    return answer
            finally:
                unit._quality_solver_aux_context_active = previous
                entry["seconds"] = time.perf_counter()-begin
                print("ATTEMPT",unit.unit_id,entry.get("residual_norm"),entry.get("jacobian_evaluations"),flush=True)
        self.patches.append(patch.object(EquilibriumStageColumnMixin,"_sparse_newton_solve",solver))
        for item in self.patches:
            item.start()
        return self

    def __exit__(self,*exc):
        for item in reversed(self.patches):
            item.stop()


def worker(args):
    import numpy as np
    import scipy.optimize  # noqa: F401 -- warm numerical machinery before solve timing
    random.seed(20261008)
    np.random.seed(20261008)
    base,inlets = normal_case(args.worker) if args.worker in NORMAL_CASES else prepare_case(args.worker)
    if args.audit_directory:
        audit_worker(args,base,inlets)
        return
    metadata = candidate_metadata(base,inlets["feed"]) if args.worker not in NORMAL_CASES else None
    with (args.output/(args.worker+"-results.jsonl")).open("x") as handle:
        for factor in args.factors if metadata else [None]:
            for initializer in args.initializers if metadata else [None]:
                params = dict(base.params,mesh_tolerance=1e-6,acceptable_mesh_residual=1e-6,
                              max_iterations=args.budget,max_jacobian_evaluations=args.budget,
                              finite_difference_rel_step=1e-6,colored_jacobian_fallback=False)
                if metadata:
                    params.update(D_to_F=factor*metadata["first_azeotrope"]["cut_capacity"],initializer=initializer)
                unit = type(base)(base.unit_id,base.thermo,params)
                if args.stage_continuation:
                    record = continue_stages(base,inlets,params,args,factor,initializer)
                    handle.write(json.dumps(record,default=json_default)+"\n")
                    handle.flush()
                    continue
                attempts = []
                record = {"case":args.worker,"factor":factor,"initializer":initializer,"params":params,"attempts":attempts}
                begin = time.perf_counter()
                print("START",args.worker,factor,initializer,flush=True)
                try:
                    with RuntimePatch(args.budget,attempts,args.mode):
                        answer = unit.solve(inlets)
                    record.update(success=True,residual=answer.performance["mesh_residual"],
                                  performance=answer.performance,warnings=answer.warnings,
                                  outputs={p:stream_record(s) for p,s in answer.outlet_streams.items()})
                    record.update(validate_output(unit,inlets,answer,phase_checks=metadata is not None))
                except Exception as error:
                    record.update(success=False,error=f"{type(error).__name__}: {error}")
                    traceback.print_exc()
                if attempts and "residual" not in record:
                    record["residual"] = attempts[-1].get("residual_norm")
                record["solve_seconds"] = time.perf_counter()-begin
                record["total_jacobians"] = sum(a.get("jacobian_evaluations",0) or 0 for a in attempts) if all(a.get("jacobian_evaluations") is not None for a in attempts) else None
                record["total_native_residual_evaluations"] = sum(a.get("function_evaluations",0) for a in attempts)
                handle.write(json.dumps(record,default=json_default)+"\n")
                handle.flush()
                print("DONE",args.worker,factor,initializer,record["success"],record.get("residual"),record["solve_seconds"],flush=True)


def audit_worker(args,base,inlets):
    import numpy as np
    from equilibrium_stage_column import EquilibriumStageColumnMixin
    records = [json.loads(line) for line in (args.audit_directory/(args.worker+"-results.jsonl")).read_text().splitlines()]
    with (args.output/(args.worker+"-audits.jsonl")).open("x") as handle:
        for record in records:
            if not record["success"]:
                continue
            unit = type(base)(base.unit_id,base.thermo,record["params"])
            attempt = record["attempts"][-1]
            vector = np.array(attempt["native_vector"])
            def frozen(self,residual,sparsity,x0,options,jacobian=None,step_event=None):
                norm = float(np.linalg.norm(residual(vector),ord=np.inf))
                return dict(x=vector,success=norm<options["mesh_tolerance"],residual_norm=norm,
                            iterations=0,jacobian_evaluations=0,function_evaluations=1,
                            jacobian_method="recorded_vector_audit",message="recorded solution")
            with patch.object(EquilibriumStageColumnMixin,"_sparse_newton_solve",frozen):
                answer = unit.solve(inlets)
            checks = validate_output(unit,inlets,answer,phase_checks=args.worker not in NORMAL_CASES)
            result = {"case":args.worker,"factor":record["factor"],"initializer":record["initializer"],
                      "residual":answer.performance["mesh_residual"],**checks,
                      "outputs":{p:stream_record(s) for p,s in answer.outlet_streams.items()}}
            handle.write(json.dumps(result,default=json_default)+"\n")
            handle.flush()
            print("AUDIT",args.worker,record["initializer"],result["residual"],checks["component_balance_error"],flush=True)


def continued_profile(unit, previous, stages, feed_stage, old_feed):
    import numpy as np
    from unit_operations_distillation import RigorousDistillation
    old_stages = len(previous["stage_liquid_compositions"])
    old_grid = np.arange(old_stages)
    mapping = np.concatenate((np.linspace(0.,old_feed-1,feed_stage),
        np.linspace(old_feed-1,old_stages-1,stages-feed_stage+1)[1:]))
    comps = list(previous["stage_liquid_compositions"][0])
    old_x = np.array([[x[c] for c in comps] for x in previous["stage_liquid_compositions"]])
    logs = np.column_stack([np.interp(mapping,old_grid,np.log(np.maximum(old_x[:,ci],1e-30))) for ci in range(len(comps))])
    weights = np.exp(logs-np.max(logs,axis=1)[:,None])
    weights /= np.sum(weights,axis=1)[:,None]
    compositions = [dict(zip(comps,row)) for row in weights]
    pressures = unit._pressure_profile(stages,float(unit.get_param("P_condenser")))
    Tmin,Tmax = unit._temperature_bounds(comps)
    temperatures = [unit._bubble_temperature_from_equation(x,P,Tmin,Tmax) for x,P in zip(compositions,pressures)]
    L = np.exp(np.interp(mapping,old_grid,np.log(previous["liquid_flows"])))
    V = np.exp(np.interp(mapping,old_grid,np.log(previous["vapor_flows"])))
    V[1] = L[0]+V[0]
    seed = {"T":temperatures,"x":compositions,"L":L,"V":V,
            "Q_cond":previous["condenser_duty_kW"]*3600.,"Q_reb":previous["reboiler_duty_kW"]*3600.,
            "initializer":"experimental_stage_continuation"}
    original = RigorousDistillation._initial_guess
    def wrapped(self,*args,**kwargs):
        return seed if self is unit else original(self,*args,**kwargs)
    return patch.object(RigorousDistillation,"_initial_guess",wrapped)


def continue_stages(base,inlets,params,args,factor,initializer):
    from unit_operations_distillation import RigorousDistillation
    target = int(params["N_stages"])
    target_feed = int(params["feed_stage"])
    previous,old_feed = None,None
    stage_records = []
    begin = time.perf_counter()
    result = {"case":args.worker,"factor":factor,"initializer":initializer,
              "stage_records":stage_records,"success":False}
    stages = list(range(4,target,args.stage_increment))+[target]
    for N in stages:
        feed_stage = 1+round((target_feed-1)*(N-1)/(target-1))
        unit = RigorousDistillation(base.unit_id,base.thermo,dict(params,N_stages=N,feed_stage=feed_stage))
        attempts = []
        item = {"N":N,"feed_stage":feed_stage,"attempts":attempts}
        stage_records.append(item)
        start = time.perf_counter()
        print("START stage",args.worker,factor,initializer,N,flush=True)
        try:
            profile_patch = continued_profile(unit,previous,N,feed_stage,old_feed) if previous else nullcontext()
            with profile_patch,RuntimePatch(args.budget,attempts,args.mode):
                answer = unit.solve(inlets)
            item.update(success=True,residual=answer.performance["mesh_residual"],performance=answer.performance)
            previous,old_feed = answer.performance,feed_stage
            if N==target:
                result.update(success=True,residual=answer.performance["mesh_residual"],performance=answer.performance,
                              outputs={p:stream_record(s) for p,s in answer.outlet_streams.items()},warnings=answer.warnings)
                total = sum(s.F for s in inlets.values())
                comps = set().union(*(s.composition for s in inlets.values()))
                result["component_balance_error"] = max(abs(sum(s.F*s.composition.get(c,0.) for s in inlets.values())-
                    sum(s.F*s.composition.get(c,0.) for s in answer.outlet_streams.values()))/max(total,1.) for c in comps)
        except Exception as error:
            item.update(success=False,error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            break
        finally:
            item["seconds"] = time.perf_counter()-start
            checkpoint = args.output/f"{args.worker}-factor{factor}-init{initializer}-stages{N}.json"
            with checkpoint.open("x") as checkpoint_handle:
                json.dump(item,checkpoint_handle,default=json_default)
            print("DONE stage",args.worker,factor,initializer,N,item.get("success"),item.get("residual"),flush=True)
    result["solve_seconds"] = time.perf_counter()-begin
    result["total_jacobians"] = sum(a.get("jacobian_evaluations",0) or 0 for s in stage_records for a in s["attempts"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",nargs="+",choices=CASES,default=["ethanol_water","ipa_water"])
    parser.add_argument("--factors",nargs="+",type=float,default=[1.02])
    parser.add_argument("--initializers",nargs="+",choices=("cheap_estimate","azeotropic"),default=["cheap_estimate","azeotropic"])
    parser.add_argument("--budget",type=int,default=250)
    parser.add_argument("--worker",choices=CASES)
    parser.add_argument("--summarize-only",action="store_true",help="Write a new summary from existing result files without solving")
    parser.add_argument("--audit-directory",type=Path,help="Validate saved successful native vectors without new nonlinear solves")
    parser.add_argument("--stage-continuation",action="store_true")
    parser.add_argument("--stage-increment",type=int,default=2)
    parser.add_argument("--mode",choices=("native","portfolio","component_flows","projection","inventory","equilibrium","coordinate_cap","box_trust","region","equilibrium_region","inside_out","inside_out_region","stable_log","stable_log_region","mass_sweep","mass_sweep_plain","material_root","material_hybr"),default="component_flows")
    args = parser.parse_args()
    if args.budget<1 or args.stage_increment<1 or any(not math.isfinite(f) or f<=0 for f in args.factors):
        parser.error("budget, stage increment, and finite cut factors must be positive")
    if args.summarize_only:
        summarize(args.output)
        return
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"probe_script.py").write_bytes(Path(__file__).read_bytes())
    (args.output/"manifest.json").write_text(json.dumps({"args":vars(args),"python":sys.version,
        "sources":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            "equilibrium_stage_column.py","unit_operations_distillation.py","unit_operations_separation.py",
            "scripts/performance/probe_azeotropic_overdistillation.py","scripts/performance/benchmark_column_trust_region.py",
            "scripts/performance/diagnose_binary_azeotropic_overdraw.py","thermodynamics_models/activity.py")}},indent=2,default=str)+"\n")
    for name in ("probe_azeotropic_overdistillation.py","benchmark_column_trust_region.py","diagnose_binary_azeotropic_overdraw.py"):
        (args.output/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    env = dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONHASHSEED="0")
    for case in args.cases:
        command = [sys.executable,str(Path(__file__).resolve()),"--output",str(args.output),"--worker",case,
                   "--budget",str(args.budget),"--mode",args.mode,"--factors",*map(str,args.factors),"--initializers",*args.initializers]
        if args.stage_continuation:
            command.extend(["--stage-continuation","--stage-increment",str(args.stage_increment)])
        if args.audit_directory:
            command.extend(["--audit-directory",str(args.audit_directory)])
        with (args.output/(case+".log")).open("x") as log:
            completed = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        print("CASE",case,completed.returncode,flush=True)
        if completed.returncode:
            raise SystemExit(f"worker failed ({completed.returncode}); see {case}.log")
    summarize(args.output)


def summarize(directory):
    records = [json.loads(line) for path in sorted(directory.glob("*-results.jsonl")) for line in path.read_text().splitlines()]
    lines = ["# Runtime coordinate/projection experiments", "",
             "Success requires the original native MESH tolerance and normal unit validation.", "",
             "| Case | Cut / capacity | Initializer | Success | Native residual | Jacobians | Native residual calls | Seconds |",
             "|---|---:|---|---|---:|---:|---:|---:|"]
    for row in records:
        attempts = row.get("attempts",[])
        norm = row.get("residual",attempts[-1].get("residual_norm") if attempts else None)
        residual = f"{norm:.3e}" if norm is not None else "unavailable"
        functions = row.get("total_native_residual_evaluations",sum(a.get("function_evaluations",0) for a in attempts))
        lines.append(f"| {row['case']} | {row['factor']} | {row['initializer']} | {row['success']} | {residual} | "
                     f"{row['total_jacobians']} | {functions} | {row['solve_seconds']:.4f} |")
        if row.get("stage_records"):
            lines.extend(["", "Stages: "+", ".join(f"{s['N']}={s['success']}" for s in row["stage_records"]), ""])
    with (directory/"summary.md").open("x") as handle:
        handle.write("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
