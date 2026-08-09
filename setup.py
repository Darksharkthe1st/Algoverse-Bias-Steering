from setuptools import setup, find_packages

setup(
    name="algo-neutrality",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "datasets",
        "huggingface-hub",
        "ipython",
        "jupyter",
        "plotly",
        "transformer-lens",
        "pandas",
        "torch",
        # bias_steer runtime deps (Phases 0-4)
        "openai",        # LLM-as-a-judge (judge.py)
        "safetensors",   # tensor persistence (artifacts.py)
        "tqdm",          # CLI progress bars (cli.py)
    ],
    # 3.10 is what the Lambda box ships; the package uses no 3.11+ syntax
    # (PEP 604 unions are 3.10) and the full test suite passes there.
    python_requires=">=3.10",
) 