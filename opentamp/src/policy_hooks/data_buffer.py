import numpy as np
import torch


MAX_BUFFER = 40000
MIN_BUFFER = 1000


class DataBuffer(object):
    def __init__(self,
                 policy,
                 sizes={},
                 default_size=MAX_BUFFER,
                 val_ratio=0.1,
                 normalize=False,
                 x_idx=None,
                 min_buffer=MIN_BUFFER,
                 ratios=None,
                 to_torch=False):
        self.sizes = sizes
        self.default_size = default_size
        self.val_ratio = val_ratio
        self.policy = policy
        self.ratios = ratios
        self.to_torch = to_torch

        self.x_idx = x_idx
        self.normalize = normalize
        self.scale = None
        self.bias = None
        self._normalized = False
        self.min_buffer = min_buffer
        self._min_sample = min_buffer // 2

        self.cur_len = 0
        self.obs = {}
        self.mu = {}
        self.wt = {}
        self.prc = {}
        self.xs = {}
        self.aux = {}
        self.x = {}
        self.primobs = {}
        self.tasks = {}
        self.lens = {}


    def update(self, data, val_ratio=0.1):
        obs, mu, prc, wt, aux, primobs, x, task, label = data
        dO = obs.shape[1:]
        dU = mu.shape[1:]
        dP = prc.shape[1:]
        dW = wt.shape[1:]
        dX = x.shape[1:] if len(x) else None
        dPrim = primobs.shape[1:] if len(primobs) else None
   
        assert label.find('VAL') < 0
        if np.random.uniform() < self.val_ratio:
            label = 'VAL_' + label

        if label not in self.lens: self.add_label(label, dO, dU, dP, dW, dPrim, dX)
        if self._normalized: obs = self.center(obs)

        for i in range(len(obs)):
            primpt = primobs[i] if len(primobs) else []
            xpt = x[i] if len(x) else []
            auxpt = aux[i].copy() if len(aux) else []
            self.add_pt(obs[i], mu[i], prc[i], wt[i], auxpt, primpt, xpt, task, label, normalize=False)

        if self.get_size() >= self.min_buffer and not self._normalized:
            self._set_scale(min(self.min_buffer, 512))
            self._normalize()
   

    def add_pt(self, obs, mu, prc, wt, aux, primobs, x, task, label, normalize=True):
        if self.lens[label] < len(self.obs[label]):
            ind = self.lens[label]
            self.aux[label].append(aux)
            self.tasks[label].append(task)
            self.lens[label] += 1
            if label.find('VAL') < 0:
                 self.cur_len += 1
        else:
            ind = np.random.randint(self.lens[label])
            self.aux[label][ind] = aux
            self.tasks[label][ind] = task
           
        if normalize: obs = self.center(obs)
        self.obs[label][ind] = obs[:]
        self.mu[label][ind] = mu[:]
        self.prc[label][ind] = prc[:]
        self.wt[label][ind] = wt
        if len(primobs): self.primobs[label][ind] = primobs[:]
        if len(x): self.x[label][ind] = x[:]
        

    def add_label(self, label, dO, dU, dP, dW, dPrim, dX):
        if label in self.lens: return

        self.lens[label] = 0
        base_label = label.split('VAL_')[-1]
        size = self.sizes.get(base_label, self.default_size)
        if base_label not in self.sizes:
            p = [self.ratios[l] for l in self.ratios]
            perc = self.ratios[base_label] / np.sum(p)
            size = int(MAX_BUFFER * perc)

        if label.find('VAL_') >= 0:
            size = int(self.val_ratio * size)

        size = max(size, self._min_sample)
        print('ADDED LABEL', label, 'WITH BUFFER SIZE', size)

        self.tasks[label] = []
        if self.to_torch:
            self.obs[label] = torch.zeros((size,)+dO, dtype=torch.float32)
            self.mu[label] = torch.zeros((size,)+dU, dtype=torch.float32)
            self.prc[label] = torch.zeros((size,)+dP, dtype=torch.float32)
            self.wt[label] = torch.zeros((size,)+dW, dtype=torch.float32)
            self.aux[label] = []
            self.primobs[label] = None
            if dPrim is not None:
                self.primobs[label] = torch.zeros((size,)+dPrim, dtype=torch.float32)
            self.x[label] = torch.zeros((size,)+dX, dtype=torch.float32)
        else:
            self.obs[label] = np.nan * np.zeros((size,)+dO, dtype=np.float32)
            self.mu[label] = np.nan * np.zeros((size,)+dU, dtype=np.float32)
            self.prc[label] = np.nan * np.zeros((size,)+dP, dtype=np.float32)
            self.wt[label] = np.nan * np.zeros((size,)+dW, dtype=np.float32)
            self.aux[label] = []
            self.primobs[label] = None
            if dPrim is not None:
                self.primobs[label] = np.zeros((size,)+dPrim, dtype=np.float32)
            self.x[label] = np.zeros((size,)+dX, dtype=np.float32)

    
    def random_label(self, val=False, min_size=0):
        min_buf = self._min_sample if not val else self._min_sample // 10
        min_size = max(min_buf, min_size)
        lab_f = lambda l: (not val and l.find('VAL_') < 0) or (val and l.find('VAL_') >= 0)
        labels = [l for l in self.lens.keys() if lab_f(l) and self.lens[l] >= min_size]
        base_labels = labels if not val else [l.split('VAL_')[-1] for l in labels]
        if not len(labels): return None

        if self.ratios is None:
            norm = np.sum([self.lens[l] for l in labels])
            p = [float(self.lens[l])/norm for l in labels]
        else:
            p = [self.ratios[l] for l in base_labels]
            for ind, l in enumerate(labels):
                if self.lens[l] < min_size:
                    p[ind] = 0.

        norm = np.sum(p)
        if norm < 1e-3: return labels[0]
        p = [pt/norm for pt in p]
        return np.random.choice(labels, p=p)


    def get_pt(self, label=None, val=False):
        if label is None: label = self.random_label(val)
        if label is None: return None

        ind = np.random.randint(self.lens[label])
        obs = self.obs[label][ind]
        mu = self.mu[label][ind]
        prc = self.prc[label][ind]
        wt = self.wt[label][ind]
        x = self.x[label][ind]
        primobs = self.primobs[label][ind] if label in self.primobs else []
        aux = self.aux[label][ind]
        return (obs, mu, prc, wt, x, primobs, aux)


    def get_batch(self, batch_size, label=None, val=False, mix=False):
        if self.get_size() < self.min_buffer: return None
        if label is None: label = self.random_label(val, min_size=batch_size)
        if label is None or self.lens.get(label, 0) < batch_size: return None

        if type(batch_size) is int:
            inds = np.random.choice(range(self.lens[label]), batch_size, replace=False)
        else:
            # Assume batch_size is a slice object or similar
            inds = batch_size

        obs = self.obs[label][inds]
        mu = self.mu[label][inds]
        prc = self.prc[label][inds]
        wt = self.wt[label][inds]
        x = self.x[label][inds]
        primobs = []
        if self.primobs[label] is not None:
            primobs = self.primobs[label][inds]
        aux = [self.aux[label][ind] for ind in inds]
        return (obs, mu, prc, wt, x, primobs, aux)


    def get_size(self, label=None, val=False):
        if label is not None:
            if val: label = 'VAL_' + label
            return self.lens[label]
        lab_f = lambda l: (not val and l.find('VAL_') < 0) or (val and l.find('VAL_') >= 0)
        min_size = self._min_sample if not val else self.val_ratio * self._min_sample
        labels = [l for l in self.lens.keys() if lab_f(l) and self.lens[l] >= min_size]

        return sum([self.lens[l] for l in labels])


    def _set_scale(self, N):
        if self.scale is not None: return self.scale, self.bias
        print('SETTING SCALE AND BIAS')
        if self.x_idx is None:
            self.x_idx = range(list(self.obs.values())[0].shape[0])

        self.scale = np.eye(len(self.x_idx))
        self.bias = np.zeros(len(self.x_idx))
        if self.normalize:
            obs = []
            for _ in range(N):
                label = self.random_label()
                ind = np.random.randint(self.lens[label])
                obs.append(self.obs[label][ind][self.x_idx])
            obs = np.array(obs)
            self.scale = np.diag(1.0 / np.maximum(np.std(obs, axis=0), 1e-1))
            self.bias = -np.mean(obs.dot(self.scale), axis=0)

        if self.policy is not None:
            self.policy.scale = self.scale
            self.policy.bias = self.bias

        return self.scale, self.bias

    
    def _normalize(self):
        if self._normalized or self.scale is None or not self.normalize:
            self._normalized = True
            return

        print('CENTERING FULL DATA...')
        for lab in self.lens:
            obs = self.obs[lab][:self.lens[lab]]
            obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias
            self.obs[lab][:self.lens[lab]] = obs
        self._normalized = True
        print('CENTERED FULL DATA')


    def center(self, obs):
        if not self.normalize: return obs
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias
        return obs


    def load_from_dir(self, dname, task):
        fnames = os.listdir(dname+'/samples/')
        fnames = list(filter(lambda f: f.find(task) >= 0, fnames))
        start_t = time.time()
        print('LOADING DATA FROM {} FILES'.format(len(fnames)))
        for fname in fnames:
            full_fname = dname+'/samples/'+fname
            data = np.load(full_fname, allow_pickle=True)
            data = data[None][0]
            self.update(data)
        print('TIME TO LOAD:', time.time() - start_t, self.get_size())


    def write_data(self, save_dir, task, n_data):
        lab = 'optimal'
        ind = np.random.choice(range(self.lens[lab]-n_data), replace=False)
        obs = self.obs[lab][ind]
        mu = self.mu[lab][ind]
        wt = self.wt[lab][ind]
        prc = self.prc[lab][ind]
        x = [self.x[lab][i] for i in ind]
        aux = [self.aux[lab][i] for i in ind]
        primobs = [self.primobs[lab][i] for i in ind]
        data = (obs, mu, prc, wt, primobs, x, aux)
        val = 'val' if np.random.uniform() <= 0.1 else 'train'
        np.save('{}/{}_{}data_{}.npy'.format(save_dir, task, val, self.cur_save), data)
        self.cur_save += 1


    def __len__(self):
        return self.cur_len


