# asyncio is Python's standard library for writing asynchronous (non-blocking) code.
# ADK's runner uses async/await, so we need this to run async functions.
import asyncio

# Agent is the core class from Google ADK used to define an AI agent.
# It binds together a model, a name, and instructions/description.
from google.adk.agents import Agent

# InMemoryRunner is a lightweight runner provided by ADK for local development and testing.
# It runs the agent in-process without needing a server or cloud deployment.
from google.adk.runners import InMemoryRunner

# load_dotenv reads the .env file and loads key=value pairs as environment variables.
# This lets us keep secrets like API keys out of the source code.
from dotenv import load_dotenv

# Load environment variables from the .env file (e.g., GOOGLE_API_KEY).
# Must be called before the Agent is initialized so the SDK can authenticate.
load_dotenv()

# Create an instance of Agent — this defines our AI agent's identity and behavior.
flight_agent = Agent(
    # The Gemini model to use for generating responses.
    # "gemini-2.5-flash" is a fast, cost-efficient Gemini model.
    model="gemini-2.5-flash",

    # A unique name for this agent. Used internally by ADK for routing and identification.
    name="FlightAgent",

    # A natural language description of what this agent does.
    # This guides the model's behavior and helps ADK understand the agent's purpose.
    description="Tell me the optimal route between flights",
)


# main() is an async function because InMemoryRunner.run_debug() is a coroutine (async).
# All ADK runner interactions must be awaited inside an async context.
async def main():
    # InMemoryRunner wraps the agent and handles the execution lifecycle locally.
    # It manages conversation state, tool calls, and model interactions in memory.
    runner = InMemoryRunner(agent=flight_agent)

    # run_debug() sends a user message to the agent and returns a stream of events.
    # It also prints debug information about each step the agent takes.
    # The string argument is the user's input prompt/question.
    events = await runner.run_debug("How many layovers are there between New York and Tokyo?")
    print(events)


# Standard Python entry point guard.
# Ensures main() only runs when this file is executed directly (not when imported as a module).
if __name__ == "__main__":
    # asyncio.run() starts the event loop, executes the async main() function,
    # and cleanly shuts down the loop when done.
    asyncio.run(main())
