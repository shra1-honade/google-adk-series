import asyncio
import uuid
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

load_dotenv()

# ── Agent ──────────────────────────────────────────────────────────────────────

# The instruction tells the agent about the structured state fields we'll be using.
# This makes the agent "state-aware" — it understands destination, budget, travel_dates
# are meaningful slots it should fill and refer back to.
root_agent = LlmAgent(
    name="travel_assistant",
    model="gemini-2.5-flash",
    instruction="""You are a helpful travel assistant.

When users provide information, acknowledge and remember:
- destination: Where they want to go
- budget: Their travel budget
- travel_dates: When they want to travel

Use stored information to give personalised recommendations.""",
)

# ── Session Service ────────────────────────────────────────────────────────────

# InMemorySessionService holds all sessions (conversations) in RAM.
# Each session has a .state dict — a key-value store for structured data
# that lives alongside the conversation history.
session_service = InMemorySessionService()

# ── Runner ─────────────────────────────────────────────────────────────────────

runner = Runner(
    agent=root_agent,
    app_name="travel_assistant",
    session_service=session_service,
)


# ── State helpers ──────────────────────────────────────────────────────────────

def store_preference(session, key: str, value: str) -> str:
    """Write a key-value pair into session.state.

    session.state is a plain Python dict attached to the session object.
    Writing here persists the value for the lifetime of the session.
    """
    session.state[key] = value
    return f"Stored: {key} = {value}"


def get_preference(session, key: str) -> str:
    """Read a value from session.state by key.

    Returns "Not set" if the key doesn't exist yet —
    avoids KeyError and makes it safe to call at any point.
    """
    return session.state.get(key, "Not set")


def show_all_preferences(session) -> dict:
    """Return the entire session.state as a plain dict.

    Useful for debugging — lets you see everything stored so far.
    """
    return dict(session.state)


# ── Chat helper ────────────────────────────────────────────────────────────────

async def chat(user_id, session_id, user_message):
    """Send one message and print the agent's reply."""
    print(f"\nYou  : {user_message}")

    message = types.Content(
        role="user",
        parts=[types.Part(text=user_message)]
    )

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response():
            print(f"Agent: {event.content.parts[0].text}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    APP_NAME   = "travel_assistant"
    USER_ID    = "traveller_01"
    SESSION_ID = str(uuid.uuid4())

    # Create the session — starts with an empty state dict and no history.
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    # Fetch the session object so we can read/write its .state dict directly.
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    print(f"Session ID: {SESSION_ID}")
    print("=" * 60)

    # ── Step 1: Manually store structured state before the conversation ────────
    # This simulates pre-filling known user data (e.g. from a profile or form).
    # We write directly into session.state — the agent can use this as context.
    print(store_preference(session, "destination",   "Japan"))
    print(store_preference(session, "budget",        "$3000"))
    print(store_preference(session, "travel_dates",  "October 2025"))

    # ── Step 2: Inspect what's in state ───────────────────────────────────────
    print("\nAll stored preferences:", show_all_preferences(session))

    # ── Step 3: Read individual values ────────────────────────────────────────
    print("Destination :", get_preference(session, "destination"))
    print("Budget      :", get_preference(session, "budget"))
    print("Hotel style :", get_preference(session, "hotel_style"))  # not set — returns "Not set"

    print("=" * 60)

    # ── Step 4: Have a conversation — agent uses the state context ─────────────
    # The agent's instructions make it refer to destination/budget/travel_dates
    # when giving recommendations, so these turns feel personalised.
    await chat(USER_ID, SESSION_ID, "What can I do in my destination?")
    await chat(USER_ID, SESSION_ID, "Is my budget enough for a good hotel?")
    await chat(USER_ID, SESSION_ID, "What should I pack for that time of year?")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
