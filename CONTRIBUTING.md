# Contributing to HDNA Workbench

Thanks for your interest in contributing.

## Getting Started

```bash
git clone https://github.com/ChrisBuilds/hdna-workbench.git
cd hdna-workbench
pip install -e ".[dev]"
```

## Running Demos

```bash
python demo.py            # PyTorch inspection
python demo_hdna.py       # HDNA core engine
python demo_adapters.py   # All adapters
python demo_tools.py      # All research tools
python demo_curricula.py  # Built-in curricula
```

## Code Style

- Python 3.9+ compatible
- Use type hints for public APIs
- Keep numpy as the only required dependency for `workbench.core`
- PyTorch is optional (only for `workbench.inspectable`)

## Adding a New Daemon

1. Subclass `workbench.core.daemon.Daemon`
2. Implement `reason(self, state, features, rng) -> Proposal | None`
3. Add to `workbench.tools.daemon_studio.TEMPLATES` if it's general-purpose
4. Write a test

## Adding a New Adapter

1. Subclass `workbench.adapters.protocol.ModelAdapter`
2. Implement `predict()`, `get_info()`, `capabilities()`
3. Implement optional methods based on what your framework supports
4. Add to `workbench/adapters/__init__.py`

## Adding a New Inspectable Layer

1. Create `workbench/inspectable/your_layer.py`
2. Subclass both the PyTorch layer and `InspectableMixin`
3. Implement `forward()` using `self._trace_forward()`
4. Implement `from_standard()` class method
5. Add to the conversion map in `workbench/inspectable/convert.py`

## Reporting Issues

Open an issue at https://github.com/ChrisBuilds/hdna-workbench/issues

## License

By contributing, you agree that your contributions will be licensed under the same [Business Source License 1.1](LICENSE) as the project.
