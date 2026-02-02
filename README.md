# Minimal Agent

A minimal observe → decide → act Python agent with a stubbed LLM call.

## Run

```bash
make run
```

## Requirements

- Python 3.9+ (no third-party dependencies)

## Configuration

- Change the goal/context in `agent.py` or instantiate `Agent` elsewhere.
- Set `max_steps` in `run()` to cap the loop.

## Project Structure

- `agent.py`: agent loop, LLM stub, JSON-only output
- `Makefile`: run target

## Notes

- Update `call_llm()` in `agent.py` to integrate a real model.
- The agent returns structured JSON output only.
