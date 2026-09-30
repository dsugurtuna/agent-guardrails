# Contributing

Thanks for taking an interest. This is a small project, so the process is light.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,anthropic]"
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
mypy            # strict
pytest          # offline; no API keys needed
python examples/email_calendar_assistant.py
```

## Expectations

- Every new control needs tests, including the failure path. Security-relevant
  behaviour (anything that decides whether a tool runs) should also have a
  concurrency or property-based test where that makes sense.
- Fail closed: when in doubt, block and explain why in the `Outcome` message.
- Explain design decisions in `docs/WHY.md` ("Why X? Because Y.") and update
  `docs/THREAT-MODEL.md` if the change affects what is or is not defended against.
- Keep examples generic and synthetic. No real personal data, no real addresses.
- Tests must not call external services. Use the fake client pattern in
  `tests/test_claude_adapter.py`.

## Reporting a security issue

Please do not open a public issue for a vulnerability. Contact the maintainer
privately first, using the details on [their GitHub profile](https://github.com/dsugurtuna).
