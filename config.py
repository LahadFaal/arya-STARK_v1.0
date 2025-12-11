# config.py

NUM_CLIENTS = 100
BYZ_FRACTION = 0.2
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.1

INPUT_DIM = 10
OUTPUT_DIM = 1

# Paths to Rust binaries (Winterfell)
WINTERFELL_PROVER_BIN = "./target/release/prove_update"
WINTERFELL_VERIFIER_BIN = "./target/release/verify_update"

# Exchange directory with Rust
TRACE_DIR = "artifacts/traces"
PROOF_DIR = "artifacts/proofs"

# PQ signature algorithm (liboqs-python)
OQS_SIG_ALG = "ML-DSA-65"  # check via oqs.get_enabled_sig_mechanisms()

DEVICE = "cpu"
RANDOM_SEED = 42
