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
    python_requires=">=3.12",
) 