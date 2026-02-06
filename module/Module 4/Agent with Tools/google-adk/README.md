# Google ADK Examples

This repository contains examples for the Google Agent Development Kit (ADK).

## Running an Agent

To run an agent, use the `google-adk` CLI.

### Suppressing Startup Warnings

The ADK may show `UserWarning` messages about experimental features on startup. To hide these, set the `PYTHONWARNINGS` environment variable to `ignore` when running your agent.

For example, to run an agent located in `2_tool/agent`:
```bash
PYTHONWARNINGS="ignore" adk agent run 2_tool/agent
```
or Run inside the folder as follows
```bash
PYTHONWARNINGS="ignore" adk agent agent
```