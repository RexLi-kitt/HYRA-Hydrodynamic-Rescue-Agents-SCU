"""Exact Gaussian conditioning on arrived records, with target-free priors.

All computations use standardized signal. The only environment-facing API
accepts legal record arrays; there is no source, target, or future-frame handle.
"""
from dataclasses import dataclass, asdict
import hashlib
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.linalg import cho_factor, cho_solve


@dataclass(frozen=True)
class Parameters:
    mean: float = .1
    scale: float = .2
    alpha: float = 0.
    kappa: float = 0.
    beta: tuple = (0., 0., 0., 0.)
    strength: float = 0.
    cross: float = 0.
    feature_mean: tuple = (0., 0., 0., 0.)
    feature_scale: tuple = (1., 1., 1., 1.)

    def validate(self):
        vals = [self.mean, self.scale, self.alpha, self.kappa, *self.beta, self.strength, self.cross,
                *self.feature_mean, *self.feature_scale]
        if not np.isfinite(vals).all() or self.scale <= 0 or not 0 <= self.alpha <= 1:
            raise ValueError('Invalid finite signal parameters')
        if not 0 <= self.kappa <= 4 or not 0 <= self.strength <= 10 or abs(self.cross) > 2:
            raise ValueError('Invalid bounded coupling parameters')
        if len(self.beta) != 4 or max(abs(v) for v in self.beta) > 2:
            raise ValueError('Invalid mean coefficients')
        if len(self.feature_mean)!=4 or len(self.feature_scale)!=4 or min(self.feature_scale)<=0:
            raise ValueError('Invalid fit-only feature normalization')


class SignalBeliefEstimator:
    def __init__(self, mode='Bpost', parameters=None, direction='forward'):
        if mode not in ('Bpost', 'R1', 'R2', 'R3', 'C1', 'C2', 'C3'):
            raise ValueError('Unknown belief mode')
        if direction not in ('forward', 'reverse', 'isotropic'):
            raise ValueError('Unknown direction diagnostic')
        self.mode, self.p, self.direction = mode, parameters or Parameters(), direction
        self.p.validate()

    def reset(self, shape, static_priors=None):
        self.shape = tuple(shape)
        if len(self.shape) != 2 or min(self.shape) < 2:
            raise ValueError('Invalid grid')
        n = int(np.prod(shape)); self.n = n
        static_priors=static_priors or {}
        self.bed=np.asarray(static_priors.get('bed',np.zeros(shape)),float)
        self.dx_m=float(static_priors.get('dx_m',5.))
        if self.bed.shape!=self.shape or not np.isfinite(self.bed).all() or not np.isfinite(self.dx_m) or self.dx_m<=0:
            raise ValueError('Invalid public bed/grid prior')
        self.bed=self.bed.ravel().copy()
        rows, cols, dx, dz = [], [], [], []
        for r in range(shape[0]):
            for c in range(shape[1]):
                for dr, dc in ((0,-1),(0,1),(-1,0),(1,0),(-1,-1),(1,-1),(-1,1),(1,1)):
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                        rows.append(r*shape[1]+c); cols.append(rr*shape[1]+cc)
                        length = np.hypot(dr, dc); dx.append(-dc/length); dz.append(-dr/length)
        self.rows, self.cols = np.asarray(rows), np.asarray(cols)
        self.dx, self.dz = np.asarray(dx), np.asarray(dz)
        rr,cc=np.divmod(self.rows,self.shape[1]); sr,sc=np.divmod(self.cols,self.shape[1])
        self.edge_distance_m=self.dx_m*np.hypot(rr-sr,cc-sc)
        self.w0 = self._matrix(np.ones(len(rows)))
        a = sparse.eye(n, format='csc') - .35*self.w0
        self.q0 = (a.T @ a).tocsc()
        self.valid = np.zeros(n, bool); self.signal = np.zeros(n); self.water = np.zeros((3,n))
        self.ages = np.zeros(n); self.obstacles = np.zeros(n, bool)
        self._cache = None
        return self

    def _matrix(self, weights):
        total = np.bincount(self.rows, weights=weights, minlength=self.n)
        values = weights / np.maximum(total[self.rows], 1e-15)
        return sparse.csc_matrix((values, (self.rows, self.cols)), shape=(self.n,self.n))

    def update(self, signal_records, water_records, validity, observed_s, now_s, observed_obstacles=None):
        mask = np.asarray(validity, bool).reshape(-1)
        signal = np.asarray(signal_records, float).reshape(-1)
        water = np.asarray(water_records, float).reshape(3,-1)
        stamps = np.asarray(observed_s, float).reshape(-1)
        if any(len(a) != self.n for a in (mask, signal, stamps)) or water.shape != (3,self.n):
            raise ValueError('Mismatched record shapes')
        if not np.isfinite(now_s) or np.any(stamps[mask] > now_s) or np.any(stamps[mask] < 0):
            raise ValueError('Unarrived or invalid observation timestamp')
        if not np.isfinite(signal[mask]).all() or not np.isfinite(water[:,mask]).all():
            raise ValueError('Nonfinite arrived measurements')
        # Unknown storage values are never used, even if the caller mutates them.
        self.valid = mask.copy(); self.signal = np.where(mask, signal, 0.)
        self.water = np.where(mask[None], water, 0.)
        self.ages = np.where(mask, now_s - stamps, 0.)
        self.obstacles = (np.asarray(observed_obstacles, bool).reshape(-1) & mask
                          if observed_obstacles is not None else np.zeros(self.n,bool))
        if self.mode.startswith('C'):
            ids = np.flatnonzero(mask)
            seed = int.from_bytes(hashlib.sha256(mask.tobytes()+np.float64(now_s).tobytes()).digest()[:8], 'little')
            order = np.random.default_rng(seed).permutation(ids)
            self.water[:,ids] = self.water[:,order]; self.ages[ids] = self.ages[order]
        self._cache = None
        return self

    def raw_features(self):
        h, u, v = self.water; speed = np.hypot(u,v)
        count = np.zeros(self.n); gradient = np.zeros(self.n)
        # A free-surface slope requires actual observations at both wet endpoints.
        legal = (self.valid[self.rows] & self.valid[self.cols] &
                 (h[self.rows]>1e-8) & (h[self.cols]>1e-8))
        elevation=self.bed+h
        np.add.at(count, self.rows[legal], 1.)
        np.add.at(gradient, self.rows[legal], np.abs(elevation[self.rows[legal]]-elevation[self.cols[legal]])/self.edge_distance_m[legal])
        gradient /= np.maximum(count, 1.)
        f = np.stack((h, speed, gradient, self.ages), axis=1)
        available=np.repeat(self.valid[:,None],4,axis=1); available[:,2]&=count>0
        return f,available

    def features(self):
        raw,available=self.raw_features()
        standardized=(raw-np.asarray(self.p.feature_mean))/np.asarray(self.p.feature_scale)
        return np.where(available,np.clip(standardized,-5.,5.),0.)

    def weights(self):
        if self.direction == 'isotropic':
            return self.w0
        _, u, v = self.water; norm = np.hypot(u,v)
        dot = (u[self.cols]*self.dx + v[self.cols]*self.dz) / np.maximum(norm[self.cols],1e-12)
        if self.direction == 'reverse': dot = -dot
        legal = self.valid[self.rows] & self.valid[self.cols] & (norm[self.cols] > 1e-8)
        fresh = np.exp(-self.ages[self.cols]/60.)
        exponent = np.where(legal, self.p.kappa*dot*fresh, 0.)
        return self._matrix(np.exp(exponent))

    def canonical(self):
        p = self.p; mode = {'C1':'R1','C2':'R2','C3':'R3'}.get(self.mode,self.mode)
        mean = np.zeros(self.n); q = self.q0
        if mode == 'R1':
            b = .35*((1-p.alpha)*self.w0 + p.alpha*self.weights())
            a = sparse.eye(self.n,format='csc')-b; q = (a.T@a).tocsc()
        if mode == 'R2': mean = self.features() @ np.asarray(p.beta)
        if mode == 'R3':
            a = sparse.hstack((-p.cross*self.weights(), sparse.eye(self.n)),format='csc')
            q = sparse.block_diag((self.q0,self.q0),format='csc') + p.strength*(a.T@a)
            mean = np.zeros(2*self.n)
        return q.tocsc(), mean

    def _condition(self):
        if self._cache is not None: return self._cache
        start = time.perf_counter(); q, mean = self.canonical(); joint = len(mean) == 2*self.n
        measured = (self.signal-self.p.mean)/self.p.scale
        known = np.r_[self.valid,self.valid] if joint else self.valid
        values = np.r_[np.where(self.obstacles,-1.,1.), measured] if joint else measured
        unknown = np.flatnonzero(~known); observed = np.flatnonzero(known)
        result = mean.copy(); result[known] = values[known]
        factor = None; residual = 0.
        if len(unknown):
            quu = q[unknown][:,unknown].tocsc()
            rhs = -q[unknown][:,observed]@(values[known]-mean[known])
            factor = splu(quu)
            shift = factor.solve(rhs); result[unknown] += shift
            residual = float(np.max(np.abs(quu@shift-rhs)))
        sig_ids = np.arange(self.n,2*self.n) if joint else np.arange(self.n)
        mapping = np.full(len(mean), -1, dtype=int); mapping[unknown] = np.arange(len(unknown))
        self._cache = (result[sig_ids]*self.p.scale+self.p.mean, factor, mapping[sig_ids],
                       dict(residual_inf=residual, seconds=time.perf_counter()-start,
                            unknown_signal=int((~self.valid).sum()), joint=joint))
        return self._cache

    def posterior(self, include_variance=False):
        mean, factor, positions, diag = self._condition()
        result = dict(mean=mean.reshape(self.shape).copy(), diagnostics=diag.copy())
        if include_variance:
            ids = np.flatnonzero(~self.valid)
            cov = self.covariance(ids)
            variance = np.zeros(self.n); variance[ids] = np.diag(cov)
            result['variance'] = variance.reshape(self.shape)
        return result

    def covariance(self, signal_indices):
        _, factor, positions, _ = self._condition()
        ids = np.asarray(signal_indices, int); k = len(ids)
        out = np.zeros((k,k)); valid = positions[ids] >= 0
        if valid.any():
            pos = positions[ids[valid]]
            rhs = np.zeros((factor.shape[0], len(pos))); rhs[pos, np.arange(len(pos))] = 1.
            out[np.ix_(valid,valid)] = factor.solve(rhs)[pos] * self.p.scale**2
        return (out+out.T)*.5

    def score(self, truth, max_density_cells=32):
        truth = np.asarray(truth).reshape(-1); pred = self.posterior()['mean'].ravel()
        ids = np.flatnonzero(~self.valid)
        if not len(ids): return dict(rmse=None, nll=None, coverage95=None, n=0)
        chosen = ids[np.linspace(0,len(ids)-1,min(len(ids),max_density_cells)).astype(int)]
        cov = self.covariance(chosen)
        chol = cho_factor(cov,lower=True,check_finite=True)
        delta = truth[chosen]-pred[chosen]
        logdet = 2*np.log(np.diag(chol[0])).sum()
        nll = .5*(delta@cho_solve(chol,delta)+logdet+len(chosen)*np.log(2*np.pi))/len(chosen)
        std = np.sqrt(np.maximum(np.diag(cov),0))
        return dict(rmse=float(np.sqrt(np.mean((truth[ids]-pred[ids])**2))),
                    nll=float(nll), coverage95=float(np.mean(np.abs(delta)<=1.96*std)), n=len(ids))
