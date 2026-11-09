"""Quantum encoders (see the PDF 'How to Encode an Image into Quantum States').

Every encoder maps a batch of classical inputs x (B, d), values in [0,1], to complex statevectors (B, 2**n) that are
computed exactly with torch (fast, GPU-friendly). Qubit q is bit q of the basis-state index (Qiskit's convention).
`qiskit_circuit` builds the equivalent real circuit for tiny examples; scripts/verify_encoders.py checks the two agree.

spec["kind"]   inputs d          qubits n                 state
amplitude      S*S pixels        2*log2(S)                |psi> = x / ||x||                       (PDF section 3)
frqi           S*S pixels        2*log2(S) + 1            (1/sqrt N) sum_i (cos t_i|0> + sin t_i|1>)|i>,  t_i = pi/2 * x_i   (FRQI)
neqr           S*S pixels        q + 2*log2(S)            (1/sqrt N) sum_i |C_i>|i>,  C_i = q-bit intensity                  (NEQR)
angle          S*S pixels        S*S (one per qubit)      prod_i Ry(pi * x_i)|0>                    (angle encoding)
zz             n features        n                        ZZ feature map (2 reps) of the winning experiments 2-4
"""
import math

import numpy as np
import torch


def _log2_int(v):
    k = int(round(math.log2(v)))
    assert 2 ** k == v, f"{v} is not a power of two"
    return k


def n_qubits_of(spec):
    k = spec["kind"]
    if k == "amplitude":
        return 2 * _log2_int(spec["size"])
    if k == "frqi":
        return 2 * _log2_int(spec["size"]) + 1
    if k == "neqr":
        return spec.get("q", 4) + 2 * _log2_int(spec["size"])
    if k == "angle":
        return spec["size"] ** 2
    return spec["n_qubits"]


def input_dim_of(spec):
    return spec["n_qubits"] if spec["kind"] == "zz" else spec["size"] ** 2


def prepare_spec(spec, x_fit=None):
    """Fill in data-dependent parts of a spec (the training-mean image for centred amplitude encoding)."""
    spec = dict(spec)
    if spec["kind"] == "amplitude" and spec.get("center") and spec.get("center_vec") is None:
        spec["center_vec"] = np.asarray(x_fit, dtype=np.float32).mean(0)
    return spec


def _hadamard_all(st, n):
    b = st.shape[0]
    r = 1.0 / math.sqrt(2.0)
    for q in range(n):
        v = st.reshape(b, 2 ** (n - 1 - q), 2, 2 ** q)
        a, c = v[:, :, 0, :], v[:, :, 1, :]
        st = torch.stack([(a + c) * r, (a - c) * r], dim=2).reshape(b, -1)
    return st


def zz_pairs(n, entanglement="linear"):
    if entanglement == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    return [(i, i + 1) for i in range(n - 1)]


def make_encoder(spec, device):
    """Returns enc(x: float32 tensor (B,d) on `device`) -> complex64 tensor (B, 2**n)."""
    kind = spec["kind"]

    if kind == "amplitude":
        cv = spec.get("center_vec")
        center = None if cv is None else torch.as_tensor(cv, dtype=torch.float32, device=device)

        def enc(x):
            if center is not None:
                x = x - center
            return (x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)).to(torch.complex64)

    elif kind == "frqi":
        def enc(x):
            th = x.clamp(0, 1) * (math.pi / 2)
            return (torch.cat([torch.cos(th), torch.sin(th)], dim=1) / math.sqrt(x.shape[1])).to(torch.complex64)

    elif kind == "neqr":
        levels = 2 ** spec.get("q", 4)

        def enc(x):
            b, npx = x.shape
            c = torch.clamp((x.clamp(0, 1) * levels).floor(), max=levels - 1).long()
            idx = c * npx + torch.arange(npx, device=x.device)[None, :]
            st = torch.zeros(b, levels * npx, device=x.device)
            st.scatter_(1, idx, 1.0 / math.sqrt(npx))
            return st.to(torch.complex64)

    elif kind == "angle":
        def enc(x):
            th = x.clamp(0, 1) * math.pi
            b, n = th.shape
            s = torch.stack([torch.cos(th / 2), torch.sin(th / 2)], dim=-1)
            out = s[:, 0, :]
            for j in range(1, n):
                out = (s[:, j, :].unsqueeze(2) * out.unsqueeze(1)).reshape(b, -1)
            return out.to(torch.complex64)

    elif kind == "zz":
        n = spec["n_qubits"]
        reps = spec.get("reps", 2)
        scale = spec.get("fm_scale", 1.0) * math.pi / 2
        pairs = zz_pairs(n, spec.get("entanglement", "linear"))
        dim = 2 ** n
        ar = torch.arange(dim, device=device)
        bits = ((ar[:, None] >> torch.arange(n, device=device)[None, :]) & 1).float()      # (D, n)
        parity = [(bits[:, i] != bits[:, j]).float() for i, j in pairs]

        def enc(x):
            x = x.clamp(0, 1) * scale
            b = x.shape[0]
            phase = 2.0 * (x @ bits.T)                                          # P(2 x_i) on every qubit
            for (i, j), par in zip(pairs, parity):                              # CX - P(2(pi-xi)(pi-xj)) - CX
                phase = phase + (2.0 * (math.pi - x[:, i]) * (math.pi - x[:, j]))[:, None] * par[None, :]
            diag = torch.exp(1j * phase)
            st = torch.zeros(b, dim, dtype=torch.complex64, device=x.device)
            st[:, 0] = 1.0
            for _ in range(reps):
                st = _hadamard_all(st, n) * diag
            return st

    else:
        raise ValueError(kind)
    return enc


# ---------------------------------------------------------------- equivalent Qiskit circuits (tiny examples only)
def qiskit_circuit(spec, x):
    """The real Qiskit circuit for one input x (1-D numpy array, values in [0,1]). Used for verification and for the
    'Methods' tab of the dashboard."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate, StatePreparation, XGate, zz_feature_map

    kind = spec["kind"]
    x = np.asarray(x, dtype=float)
    n = n_qubits_of(spec)

    if kind == "amplitude":
        qc = QuantumCircuit(n, name="Amplitude")
        qc.append(StatePreparation(x / np.linalg.norm(x)), range(n))
    elif kind == "angle":
        qc = QuantumCircuit(n, name="Angle")
        for i, v in enumerate(x):
            qc.ry(float(np.clip(v, 0, 1)) * np.pi, i)
    elif kind == "frqi":
        k = (n - 1) // 2
        qc = QuantumCircuit(n, name="FRQI")
        pos = list(range(2 * k))
        qc.h(pos)
        for i, v in enumerate(x):                                                # pixel i: Ry(2*theta_i) controlled on |i>
            qc.append(RYGate(float(np.clip(v, 0, 1)) * np.pi).control(2 * k, ctrl_state=i), pos + [2 * k])
    elif kind == "neqr":
        q = spec.get("q", 4)
        k = (n - q) // 2
        levels = 2 ** q
        qc = QuantumCircuit(n, name="NEQR")
        pos = list(range(2 * k))
        qc.h(pos)
        for i, v in enumerate(x):
            c = min(int(np.clip(v, 0, 1) * levels), levels - 1)
            for b in range(q):
                if (c >> b) & 1:
                    qc.append(XGate().control(2 * k, ctrl_state=i), pos + [2 * k + b])
    elif kind == "zz":
        fm = zz_feature_map(n, reps=spec.get("reps", 2), entanglement=spec.get("entanglement", "linear"))
        params = sorted(fm.parameters, key=lambda p: p.index)
        scale = spec.get("fm_scale", 1.0) * np.pi / 2
        qc = fm.assign_parameters(dict(zip(params, np.clip(x, 0, 1) * scale)))
    else:
        raise ValueError(kind)
    return qc
