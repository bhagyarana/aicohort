# MCP Project Setup

This guide will help you set up the MCP project by installing required packages and pulling the necessary Docker image.

## Prerequisites

- Python 3.13
- Docker installed and running
- pip package manager

## Installation Steps

### 1. Install Python Dependencies

First, create a conda environment with Python 3.13:

```bash
conda create -n mcp python=3.13 -y
conda activate mcp
```

Then, ensure you have the required Python packages installed from the requirements file:

```bash
pip install -r requirements.txt
```


### 2. Pull Docker Image

Pull the GitHub MCP server Docker image:

```bash
docker pull ghcr.io/github/github-mcp-server:latest
```