<h1 align="center">
  <a href="https://github.com/kAIto47802/rcmax-with-predictions">
    <img width="94%" height="13px" src="https://github.com/user-attachments/assets/12769944-6c48-4d6b-85a8-e457bba526a8" />
  </a>
  Learning-Augmented Approximation for Unrelated-Machines Makespan Scheduling
  <a href="https://github.com/kAIto47802/rcmax-with-predictions">
    <img width="94%" height="9px" src="https://github.com/user-attachments/assets/54d12ccb-c5fb-4a6e-aa25-d9d6d1d567e7" />
  </a>
</h1>

<p align="center">
  <strong>✨️ Accepted to the Fortieth Annual Conference on Neural Information Processing Systems ✨</strong>
  <br />
  <strong> (NeurIPS 2026) </strong>
</p>

<div  align="center">
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue" alt="Python" />
  </a>
  <a href="https://arxiv.org/abs/2606.13133">
    <img src="https://img.shields.io/badge/arXiv-2606.13133-b31b1b.svg" alt="arXiv" />
  </a>
  <a href="https://neurips.cc/virtual/2026/loc/sydney/poster/153033">
    <img src="https://img.shields.io/badge/NeurIPS-2026-68438a.svg" alt="NeurIPS 2026" />
  </a>
</div>

<br />

<h2 align="center">
  <div>📊 Reproducing the Results 📊</div>
  <a href="https://github.com/kAIto47802/rcmax-with-predictions">
    <img width="80%" height="8px" src="https://github.com/user-attachments/assets/74f2e081-8a91-4683-960d-1929e4dbdbd1" />
  </a>
</h2>

All experiments in our paper are fully reproducible.

1. Installation

```bash
uv sync --locked --python 3.13 --extra experiments
# If you're using pip:
# pip install --upgrade pip  # enable PEP 660 support
# pip install -e '.[experiments]'
```

2. Run Experiments

To run the experiments varying false negatives and false positives (Figures 1(a), 1(b), and 2):

```bash
./run_smoothness.sh
```

To run the experiments measuring the runtime (Figure 1(c) and 3):

```bash
./run_runtime.sh
```

3. Check the Results

The generated figures will be saved in the `figures/` directory. The raw experiment outputs (pickled results) and the values used in the tables are saved under `results/`.

> [!NOTE]
> If a LaTeX environment is not available, the figures will be rendered without LaTeX, which may slightly change font rendering and layout.

<h2 align="center">
  <div>🔖 Citation 🔖</div>
  <a href="https://github.com/kAIto47802/rcmax-with-predictions">
    <img width="80%" height="8px" src="https://github.com/user-attachments/assets/74f2e081-8a91-4683-960d-1929e4dbdbd1" />
  </a>
</h2>

```bibtex
@inproceedings{baba2026learning,
  title     = {Learning-Augmented Approximation for Unrelated-Machines Makespan Scheduling},
  author    = {Kaito Baba and Evripidis Bampis and Giorgos Mitropoulos},
  year      = {2026},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
}
```
