# asyncio — Python's async runtime, needed because ADK's Runner uses async/await.
import asyncio

# uuid — generates unique IDs. We use it to create a unique session ID
# so each conversation thread is uniquely identifiable.
import uuid

# load_dotenv reads the .env file and injects GOOGLE_API_KEY into the environment.
from dotenv import load_dotenv

# types from google.genai is used to structure messages in the format ADK expects.
# Content = a single message, Part = the text content inside that message.
from google.genai import types

# LlmAgent is the core agent class in ADK (same as Agent — they are aliases).
# It defines the AI's name, which model to use, and its behaviour instructions.
from google.adk.agents import LlmAgent

# InMemorySessionService stores conversation history (session) in RAM.
# It tracks what was said across multiple turns within the same session.
# Data is lost when the program exits — fine for dev/learning, not for production.
from google.adk.sessions import InMemorySessionService

# Runner is the orchestrator that connects the agent to the session service.
# Unlike InMemoryRunner (file 01), this Runner accepts a session_service,
# which enables it to load and save conversation history automatically.
from google.adk.runners import Runner

# Loads GOOGLE_API_KEY from .env so the ADK can authenticate with Gemini.
load_dotenv()

# ── Agent ─────────────────────────────────────────────────────────────────────

# LlmAgent defines the agent's identity and behaviour.
root_agent = LlmAgent(
    # Internal name used by ADK to identify this agent.
    name="travel_assistant",

    # The Gemini model that powers this agent's responses.
    model="gemini-2.5-flash",

    # System instruction — sets the agent's persona and rules.
    # This is sent to the model before every conversation turn (hidden from user).
    # Here we explicitly tell it to remember destinations and preferences,
    # which leverages the session history we're providing it.
    instruction=(
        "You are a helpful travel assistant. "
        "Remember the user's destination and preferences throughout the conversation. "
        "When they mention a destination, refer back to it in follow-up questions."
    ),
)

# ── Session Service ────────────────────────────────────────────────────────────

# InMemorySessionService is the memory manager.
# It stores a list of sessions, each containing the full message history.
# When run_async() is called, the Runner asks this service for the session's
# history and passes it to the model — that's how the agent "remembers".
session_service = InMemorySessionService()

# ── Runner ────────────────────────────────────────────────────────────────────

# Runner is the execution engine. It:
#   1. Receives a new user message
#   2. Fetches the conversation history from session_service
#   3. Sends history + new message to the agent/model
#   4. Saves the agent's reply back into the session
#   5. Streams events (including the final reply) back to the caller
runner = Runner(
    agent=root_agent,             # the agent to run
    app_name="travel_assistant",  # logical app name — groups sessions together
    session_service=session_service,  # the memory store to use
)


# ── Chat helper ───────────────────────────────────────────────────────────────

async def chat(user_id, session_id, user_message):
    """Send one message to the agent and print its reply."""
    print(f"\nYou  : {user_message}")

    # Wrap the user's plain text into the structured Content format ADK requires.
    # role="user" tells the model this is a human message (vs role="model").
    # parts is a list — a message can have multiple parts (text, images, etc).
    message = types.Content(
        role="user",
        parts=[types.Part(text=user_message)]
    )

    # run_async() sends the message through the runner and yields a stream of events.
    # Passing the same session_id each time is critical — it tells the runner
    # which conversation history to load, giving the agent its memory.
    async for event in runner.run_async(
        user_id=user_id,        # identifies the user within the app
        session_id=session_id,  # identifies which conversation thread to use
        new_message=message,    # the new message to add to that conversation
    ):
        # The runner emits multiple events per turn (thinking steps, tool calls, etc).
        # is_final_response() filters for only the last event — the agent's actual reply.
        if event.is_final_response():
            print(f"Agent: {event.content.parts[0].text}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    # A fixed string identifying this user across sessions.
    user_id = "traveller_01"

    # Generate a unique ID for this conversation session.
    # Think of it as a "conversation thread ID".
    # Reusing the same ID = same memory. New ID = fresh start.
    session_id = str(uuid.uuid4())

    # Register the session with the session service.
    # This creates an empty conversation history for this session_id.
    # Must be done before sending any messages.
    await session_service.create_session(
        app_name="travel_assistant",
        user_id=user_id,
        session_id=session_id,
    )
    print(f"Session ID: {session_id}")
    print("=" * 60)

    # ── Multi-turn conversation ────────────────────────────────────────────────
    # Each call reuses the same session_id, so the agent sees the full
    # conversation history on every turn.
    #
    # Turn 1: User establishes destination ("Japan")
    await chat(user_id, session_id, "I'm planning a trip to Japan.")

    # Turn 2: No destination mentioned — agent infers "Japan" from session history
    await chat(user_id, session_id, "What's the best time of year to visit?")

    # Turn 3: Agent builds on both previous turns to give a contextual itinerary
    await chat(user_id, session_id, "Can you suggest a 7-day itinerary?")

    print("\n" + "=" * 60)


# Run the async main() function using asyncio's event loop.
if __name__ == "__main__":
    asyncio.run(main())
