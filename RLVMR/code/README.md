# RLVMR code package

This directory contains the vendored `verl` training code and environment
interfaces used by the RLVMR project. Project documentation is in
[`../README.md`](../README.md). This local README supplies the package description
required by both `setup.py` and `pyproject.toml` for editable and wheel builds.

The ScienceWorld diagnostic experiment in this branch uses the ordinary GiGPO
entry point, without RLVMR meta-reasoning rewards or SFT. Its pinned model,
reference recipe, execution differences and Orion scripts are documented in
[`experiments/gigpo_diag/CONFIG_DIFF.md`](experiments/gigpo_diag/CONFIG_DIFF.md).

Installation and training are separate stages. A successful package build does
not establish a working GPU runtime or a completed training update.
