"""The final model set. Each module has ONE quantum encoding, trained as a quantum-kernel SVM and as a VQC."""

# ---- chest X-ray: DenseNet-121 features -> PCA (16 numbers in [0,1]) -> ZZ feature map -> 16 qubits
XRAY_Q = {
    "HYB16": dict(kind="zz", n_qubits=16, rep="PCA16", label="CNN features -> PCA 16 -> ZZ map"),
}
# classical models see the same 16 numbers (like-for-like) and, as a 'ceiling', all 1,024 CNN features
XRAY_CLASSICAL_REPS = ["PCA16", "DENSE"]
REP_LABEL = {"PCA16": "16 CNN-PCA features", "DENSE": "1,024 CNN features", "H16": "16 clinical features"}

# ---- heart disease: the 16 survey features -> ZZ feature map -> 16 qubits
HEART_Q = {
    "H16": dict(kind="zz", n_qubits=16, rep="H16", label="16 features -> ZZ map"),
}

# ---- kernel-SVM search space (the kernel is exp(gamma (F-1)/s); gamma = 0 means the raw fidelity)
KERNEL_GAMMAS = [0, 0.25, 0.5, 1, 2, 4]
KERNEL_CS = [0.1, 1, 10, 100]


def variants_for(spec):
    """Encoder variants tried on the validation set: the bandwidth (input scaling) of the ZZ feature map."""
    if spec["kind"] == "zz":
        # at 16 qubits larger scales make every pair of inputs nearly orthogonal (kernel concentration), so go small
        return [{"fm_scale": s} for s in (0.02, 0.05, 0.1, 0.3)]
    return [{}]
