from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from runner_agent.util import load_instruction_from_file
from runner_agent.agent import youtube_shorts_agent
import asyncio
# Load .env
# Replace the API_KEY in .env file.
from dotenv import load_dotenv

load_dotenv()

# Instantiate constants
APP_NAME = "youtube_shorts_app"
USER_ID = "12345"
SESSION_ID = "123344"

# Session and Runner
async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=youtube_shorts_agent, app_name=APP_NAME, session_service=session_service)
    return session, runner


# Agent Interaction
async def call_agent_async(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    session, runner = await setup_session_and_runner()
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

async def main():
    """Asynchronous main function to run the agent."""
    await call_agent_async("I want to write a short on how to build AI Agents")

if __name__ == "__main__":
    # Use asyncio.run() to execute the async main function.
    asyncio.run(main())