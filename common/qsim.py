"""Differentiable exact simulator for the trainable part of the variational quantum classifier.

Ansatz = the winner of experiments 3-4:  (reps) x [ RZ-RY-RZ rotation on every qubit  +  CNOT ring ]  + final RZ-RY-RZ block.
Readout = probability that the readout qubit(s) are measured as 1 (a local measurement, to avoid barren plateaus).
"""
import numpy as np
import torch


class Ansatz:
    def __init__(self, n, reps=2, entanglement="ring", readout="z0", device="cpu"):
        self.n, self.reps, self.dim = n, reps, 2 ** n
        pairs = [(q, q + 1) for q in range(n - 1)]
        if entanglement == "ring":
            pairs.append((n - 1, 0))
        idx = np.arange(self.dim)
        cur = idx.copy()
        for c, t in pairs:                                      # follow every basis state through the CNOT sequence
            cur = np.where(((cur >> c) & 1) == 1, cur ^ (1 << t), cur)
        src = np.empty(self.dim, dtype=np.int64)
        src[cur] = idx                                          # new_state[f] = old_state[src[f]]
        self.src = torch.from_numpy(src).to(device)
        if readout == "z0":                                     # P(qubit 0 = 1)
            mask = (idx & 1) == 1
        elif readout == "parity01":                             # parity of qubits 0 and 1
            mask = (((idx & 1) ^ ((idx >> 1) & 1)) == 1)
        else:                                                   # parity of all qubits
            mask = np.array([bin(i).count("1") % 2 for i in idx], dtype=bool)
        self.mask = torch.from_numpy(mask.astype(np.float32)).to(device)

    @property
    def n_params(self):
        return (self.reps + 1) * 3 * self.n

    @staticmethod
    def _rot(th):
        """th (3, n) -> (n, 2, 2) complex: RZ(c) RY(b) RZ(a) for every qubit."""
        a, b, c = th[0], th[1], th[2]
        ea_m, ea_p = torch.exp(-0.5j * a), torch.exp(0.5j * a)
        ec_m, ec_p = torch.exp(-0.5j * c), torch.exp(0.5j * c)
        cb = torch.cos(b / 2).to(torch.complex64)
        sb = torch.sin(b / 2).to(torch.complex64)
        u00, u01 = ec_m * cb * ea_m, -ec_m * sb * ea_p
        u10, u11 = ec_p * sb * ea_m, ec_p * cb * ea_p
        return torch.stack([torch.stack([u00, u01], -1), torch.stack([u10, u11], -1)], -2)

    def apply(self, state, theta):
        """state (B, 2^n) complex, theta (reps+1, 3, n) real -> state after the ansatz."""
        b = state.shape[0]
        for layer in range(self.reps + 1):
            u = self._rot(theta[layer])
            for q in range(self.n):
                v = state.reshape(-1, 2, 2 ** q)
                state = torch.einsum("ab,mbz->maz", u[q], v).reshape(b, self.dim)
            if layer < self.reps:
                state = state[:, self.src]
        return state

    def class1_prob(self, state):
        p = state.real ** 2 + state.imag ** 2
        return (p * self.mask).sum(1)
