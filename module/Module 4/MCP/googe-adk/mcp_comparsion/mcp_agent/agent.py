from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
import os

root_agent = Agent(
    model="gemini-2.0-flash",
    name="MCP_Agent",
    instruction="""
    You are a helpful assistant that can help with a variety of tasks.
    """,
    tools=[
        # MCPToolset(
        #     connection_params=StdioServerParameters(
        #         command="python", args=["absolute_path_to_server.py"]
        #     )
        # ),
        # https://github.com/github/github-mcp-server
        MCPToolset(
            connection_params=StdioServerParameters(
                command="docker",
                args=[
                    "run",
                    "-i",
                    "--rm",
                    "-e",
                    "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "ghcr.io/github/github-mcp-server",
                ],
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")},
            )
        ),
    ],
)
