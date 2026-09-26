"""Shared deterministic isotherm fitting tools for catalogue builders and audits."""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from physical_constants import R_J_MOL_K as R


def prediction(v, form, f, t, thermal):
    qmax, b = np.exp(v[:2])
    index = {'langmuir':2,'dual_site_langmuir':4,'toth':3,'sips':3}[form]
    heat = v[index]*1000 if thermal else 0.
    logbf = np.log(b*f)+heat/R*(1/t-1/298.15)
    if form=='toth':
        n=np.exp(v[2])
        return qmax*np.exp(logbf-np.logaddexp(0,n*logbf)/n)
    if form=='sips':
        return qmax*expit(np.exp(v[2])*logbf)
    q = qmax*expit(logbf)
    if form=='dual_site_langmuir':
        q2,b2=np.exp(v[2:4])
        heat2=v[5]*1000 if thermal else 0.
        q += q2*expit(np.log(b2*f)+heat2/R*(1/t-1/298.15))
    return q


def fit(form,f,t,q,thermal,scale):
    best=None
    # Include both the original builder starts and comparison starts. Otherwise
    # a worse local minimum can masquerade as evidence for a different form.
    for b in (0.1,1.,10.,100.,1000.,10000.):
        v=list(np.log([max(q)*1.4,b]))
        lo,hi=[-12.,-25.],[6.,35.]
        if form=='dual_site_langmuir':
            v+=list(np.log([max(q)*.7,b*.03]));lo += [-12.,-25.];hi += [6.,35.]
        elif form in {'sips','toth'}:
            v += [np.log(.7)];lo += [np.log(.05)];hi += [np.log(2.)]
        if thermal:
            v += [30.,20.] if form=='dual_site_langmuir' else [25.]
            lo += [0.,0.] if form=='dual_site_langmuir' else [0.]
            hi += [120.,120.] if form=='dual_site_langmuir' else [120.]
        result=least_squares(lambda x:(prediction(x,form,f,t,thermal)-q)/scale,
            v,bounds=(lo,hi),max_nfev=3000,ftol=1e-10,xtol=1e-10,gtol=1e-9)
        if result.success and (best is None or np.dot(result.fun,result.fun)<np.dot(best.fun,best.fun)):
            best=result
    if best is None:
        raise RuntimeError(f'No converged {form} fit')
    return best


def parameter_count(form, thermal):
    return {'langmuir':2,'dual_site_langmuir':4,'toth':3,'sips':3}[form] + (
        (2 if form=='dual_site_langmuir' else 1) if thermal else 0)


def select_fit(f,t,q,fold,baseline):
    """Select by held-out MARE, rejecting underdetermined validation folds.

    Prefer fewer parameters and a Henry-consistent form when CV errors agree
    within 1% relative. Low-pressure Sips extrapolation is flagged in metadata.
    """
    thermal=len(set(t))>1
    validation = 'interleaved three-fold within source curves'
    if min(np.count_nonzero(fold!=v) for v in set(fold)) < parameter_count('langmuir',thermal):
        fold=np.arange(len(q))
        validation = 'leave-one-out; too few points for three-fold validation'
    scale=np.maximum(q,.03*max(q))
    candidates={}
    fits={}
    for form in ('langmuir','dual_site_langmuir','toth','sips'):
        nparams=parameter_count(form,thermal)
        if any(np.count_nonzero(fold!=v)<nparams for v in set(fold)):
            candidates[form]={'eligible':False,'reason':'Too few training points for parameter count'}
            continue
        try:
            full=fit(form,f,t,q,thermal,scale)
            held=np.empty(len(q))
            for split in sorted(set(fold)):
                test=fold==split;train=~test
                trained=fit(form,f[train],t[train],q[train],thermal,np.maximum(q[train],.03*max(q[train])))
                held[test]=prediction(trained.x,form,f[test],t[test],thermal)
        except RuntimeError as error:
            candidates[form]={'eligible':False,'reason':str(error)}
            continue
        pred=prediction(full.x,form,f,t,thermal)
        candidates[form]={'eligible':True,'parameters_count':nparams,
            'validation_method':validation,
            'train_mare':float(np.mean(abs(pred-q)/q)),
            'validation_mare':float(np.mean(abs(held-q)/q)),
            'rmse_mol_kg':float(np.sqrt(np.mean((pred-q)**2)))}
        fits[form]=full
    eligible=[k for k in fits]
    if not eligible:
        raise RuntimeError('No identifiable model for held-out validation')
    best=min(candidates[k]['validation_mare'] for k in eligible)
    similar=[k for k in eligible if candidates[k]['validation_mare']<=best*1.01+1e-10]
    chosen=min(similar,key=lambda k:(parameter_count(k,thermal),k=='sips',candidates[k]['validation_mare']))
    return chosen,fits[chosen],candidates


def setting_from_fit(form,fit_result,thermal):
    v=fit_result.x
    index={'langmuir':2,'dual_site_langmuir':4,'toth':3,'sips':3}[form]
    settings={'model':form,'qmax':float(np.exp(v[0])),'b':float(np.exp(v[1])),
              'T_ref':298.15,'heat':float(v[index]*1000) if thermal else 0.}
    if form in {'toth','sips'}:
        settings['n']=float(np.exp(v[2]))
    if form=='dual_site_langmuir':
        settings.update(qmax2=float(np.exp(v[2])),b2=float(np.exp(v[3])),
                        heat2=float(v[5]*1000) if thermal else 0.)
    return settings
