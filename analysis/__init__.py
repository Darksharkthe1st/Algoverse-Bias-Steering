"""Standalone analysis of run outputs (arch roadmap §7.1).

This package reads what runs produce (`runs/index.csv` + per-run `results.csv`) and
deliberately does NOT import the `bias_steer` engine: running produces data,
analysis consumes it, and the two never re-run each other.
"""
