"""STEP 2 (seconds) - prove the fast torch encoders and the ansatz simulator equal the real Qiskit circuits.

    python scripts/verify_encoders.py

Prints PASS / FAIL per check (it never stops at the first failure) so you can see exactly what, if anything, differs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from common import encoders as E
from common.qsim import Ansatz

rng = np.random.default_rng(0)
dev = torch.device("cpu")
results = []


def check(name, fn):
    try:
        val = fn()
        ok = val > 1 - 1e-6
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: min fidelity {val:.10f}")
    except Exception as e:                                           # noqa: BLE001
        results.append(False)
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")


def encoder_check(spec, x):
    spec = dict(spec)
    spec["n_qubits"] = E.n_qubits_of(spec)
    mine = E.make_encoder(spec, dev)(torch.as_tensor(x[None], dtype=torch.float32))[0].numpy()
    ref = Statevector.from_instruction(E.qiskit_circuit(spec, x)).data
    return float(abs(np.vdot(ref, mine)) ** 2)


check("amplitude 2x2 (2 qubits)", lambda: encoder_check(dict(kind="amplitude", size=2), rng.random(4)))
check("amplitude 4x4 (4 qubits)", lambda: encoder_check(dict(kind="amplitude", size=4), rng.random(16)))
check("FRQI 2x2 (3 qubits)", lambda: encoder_check(dict(kind="frqi", size=2), rng.random(4)))
check("FRQI 4x4 (5 qubits)", lambda: encoder_check(dict(kind="frqi", size=4), rng.random(16)))
check("NEQR 2x2, 2-bit (4 qubits)", lambda: encoder_check(dict(kind="neqr", size=2, q=2), rng.random(4)))
check("NEQR 4x4, 3-bit (7 qubits)", lambda: encoder_check(dict(kind="neqr", size=4, q=3), rng.random(16)))
check("angle 2x2 (4 qubits)", lambda: encoder_check(dict(kind="angle", size=2), rng.random(4)))
check("ZZ map, 4 qubits, linear", lambda: encoder_check(dict(kind="zz", n_qubits=4, fm_scale=1.0), rng.random(4)))
check("ZZ map, 5 qubits, scale 0.3", lambda: encoder_check(dict(kind="zz", n_qubits=5, fm_scale=0.3), rng.random(5)))


def ansatz_check(n=4, reps=2, entanglement="ring"):
    ans = Ansatz(n, reps, entanglement, "z0", dev)
    theta = rng.normal(size=(reps + 1, 3, n)).astype(np.float32)
    x = rng.random(n)
    spec = dict(kind="zz", n_qubits=n, fm_scale=1.0)
    state = E.make_encoder(spec, dev)(torch.as_tensor(x[None], dtype=torch.float32))
    mine = float(ans.class1_prob(ans.apply(state, torch.as_tensor(theta)))[0])
    qc = E.qiskit_circuit(spec, x)
    for layer in range(reps + 1):
        for q in range(n):
            qc.rz(float(theta[layer, 0, q]), q)
            qc.ry(float(theta[layer, 1, q]), q)
            qc.rz(float(theta[layer, 2, q]), q)
        if layer < reps:
            for q in range(n - 1):
                qc.cx(q, q + 1)
            if entanglement == "ring":
                qc.cx(n - 1, 0)
    probs = Statevector.from_instruction(qc).probabilities()
    ref = float(sum(p for i, p in enumerate(probs) if i & 1))
    print(f"      ansatz ({entanglement}, n={n}): simulator P(q0=1) = {mine:.8f}, Qiskit = {ref:.8f}")
    return 1.0 - abs(mine - ref)


check("RZ-RY-RZ ring ansatz + readout", lambda: ansatz_check(4, 2, "ring"))
check("RZ-RY-RZ linear ansatz + readout", lambda: ansatz_check(5, 1, "linear"))

print(f"\n{sum(results)}/{len(results)} checks passed")
