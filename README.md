# Minimal Agent

A minimal observe → decide → act Python agent with explicit tools, memory, guardrails, and reflection. The LLM is a stub and can be replaced with a real model call.

## Run

```bash
make run
```

## Demo Mode

```bash
AGENT_DEMO=1 make run
```

## Requirements

- Python 3.9+ (no third-party dependencies)

## Configuration

- Change the goal/context in `agent.py` or instantiate `Agent` elsewhere.
- Set `max_steps` and guardrail budgets in the `Agent` constructor.
- Long-term memory file defaults to `long_term_memory.json` in the working directory.

## Project Structure

- `agent.py`: agent loop, tool validation, reflection
- `tools.py`: tool registry
- `memory.py`: short-term and long-term memory
- `guardrails.py`: execution and budget guardrails
- `llm.py`: LLM stub and demo mode
- `utils.py`: shared helpers
- `Makefile`: run target
