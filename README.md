# arya-STARK v1.0 — Reproducible Research Artefact

This repository provides the reference implementation and experimental artefacts for **arya-STARK**, a proof-of-concept system combining zk-STARK proofs, post-quantum authentication, and Byzantine-resilient aggregation in federated learning.

The code is intended to support **reproducibility of the experimental evaluation** reported in the accompanying paper. It is **not a production system**, but a research prototype designed to run on a standard laptop.

## 📁 Repository Structure

├── artifacts/ # Experimental logs, metrics, and result artefacts
├── aryaStark/ # zk-STARK proof generation and verification (Rust)
├── clients/ # Federated learning clients (Python)
├── server/ # Server-side verification and aggregation logic (Python)
├── models/ # Reference ML models used in experiments
├── utils/ # Shared utilities (encoding, logging, helpers)
├── config.py # Global configuration parameters
├── main.py # Entry point for running experiments
└── .gitignore


---

## ⚙️ System Requirements

The artefact has been tested on a commodity laptop with the following setup:

- **OS**: Linux or macOS (Windows via WSL2 should work)
- **CPU**: x86_64 (no GPU required)
- **RAM**: ≥ 8 GB
- **Python**: 3.9 or newer
- **Rust**: stable toolchain (for zk-STARK component)


## 🔧 Installation Instructions

### 1. Clone the repository

$bash
git clone https://github.com/ccsArtifacts/arya-STARK_v1.0.git
cd arya-STARK_v1.0

### 2. Python environment setup
It is recommended to use a virtual environment.

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

If requirements.txt is not present, install the following minimal dependencies:

pip install numpy torch tqdm matplotlib

### 3. Rust toolchain setup (for zk-STARK)

Install Rust if not already available:

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

Build the STARK prover/verifier:

cd aryaStark
cargo build --release
cd ..

### Running the Proof-of-Concept
1. Configuration

All experimental parameters (number of clients, Byzantine ratio, rounds, cryptographic parameters) are defined in:

config.py
Default configuration reproduces the results reported in the paper:

100 clients
20% Byzantine clients
5 federated rounds

### 2. Launch an experiment

From the repository root:
python main.py

This will:

1. Initialize federated learning clients

2. Perform local training and gradient encoding

3. Generate zk-STARK proofs for each client update

4. Sign proofs using post-quantum signatures

5. Verify proofs and signatures on the server

6. Apply Byzantine-resilient aggregation

7. Log performance metrics

### Experimental Outputs

All logs and metrics are stored in:
artifacts/

This includes:

1. Client-side execution times

2. Server-side verification times

3. Proof sizes and signature sizes

4. Aggregation statistics

These artefacts are used to generate the figures and tables reported in the paper.

### Reproducibility Notes

All experiments are deterministic given the same configuration.

No external services or GPUs are required.

The implementation is single-machine and intended for feasibility and overhead evaluation.

Results may vary slightly depending on CPU frequency and OS scheduling.

 ### Limitations ⚠️

This is a research prototype, not a production-ready FL system.

The server is assumed to be "honest".

Secure aggregation is out of scope for this artefact.

The implementation focuses on cryptographic verification overhead rather than ML accuracy benchmarking.

### License 📜 

This artefact is released for academic and research use only.

