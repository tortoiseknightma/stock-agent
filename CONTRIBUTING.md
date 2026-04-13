# Contributing

Contributions are welcome. Here's how to get started.

## Development Setup

```bash
git clone https://github.com/tortoiseknightma/stock-agent.git
cd stock-agent
pip install -r requirements.txt
```

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Public release — user-facing code and docs |
| `dev` | Private development — design notes, experiments |

## Making Changes

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Test with simulated broker: `python cli.py status && python cli.py analyze`
5. Commit: `git commit -m "feat: add my feature"`
6. Push and open a PR

## Code Style

- Python 3.10+ compatible
- Type hints on public APIs
- Docstrings on classes and public methods
- Dataclasses for data structures

## Adding a New Data Source

1. Create a class in `data/sources/`
2. Implement a standard interface (e.g., `get_ticker_news(ticker) -> List[NewsItem]`)
3. Register it in `data/sources/__init__.py`
4. Wire it into `agent.py` if needed

## Adding a New Analyzer

1. Create a module in `analysis/`
2. Produce a score from -1 to 1
3. Return a signal dataclass with `score`, `signal`, and `indicators`
4. Add to `CompositeAnalyzer` with a configurable weight

## Reporting Issues

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your `config.yaml` (redact any secrets)
