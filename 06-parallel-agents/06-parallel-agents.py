import asyncio
import uuid
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent, ParallelAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search

load_dotenv()

# ── Sub-agents ─────────────────────────────────────────────────────────────────
#
# Each sub-agent is a fully independent LlmAgent with its own instruction and
# tool. They share no state — ParallelAgent gives each one an isolated branch
# context so their conversation histories don't interfere with each other.

hotel_agent = LlmAgent(
    name="hotel_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a hotel specialist. "
        "Use google_search to find current hotel options, prices, and neighbourhoods. "
        "Only answer questions about accommodation."
    ),
    tools=[google_search],
)

flight_agent = LlmAgent(
    name="flight_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a flight specialist. "
        "Use google_search to find current flight routes, airlines, and booking tips. "
        "Only answer questions about flights."
    ),
    tools=[google_search],
)

sightseeing_agent = LlmAgent(
    name="sightseeing_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a sightseeing specialist. "
        "Use google_search to find current attractions, activities, and local experiences "
        "at the destination. Only answer questions about things to do."
    ),
    tools=[google_search],
)

# ── ParallelAgent ──────────────────────────────────────────────────────────────
#
# ParallelAgent wraps the three sub-agents and runs them concurrently.
# Internally it uses asyncio.TaskGroup (Python 3.11+) to launch all three
# at the same time, merging their event streams via an async queue.
#
# Key properties:
#   - All sub-agents receive the same user query simultaneously.
#   - Each runs in an isolated branch — no shared state between them.
#   - Responses arrive in non-deterministic order (whichever finishes first).
#   - event.author identifies which sub-agent produced each event.

root_agent = ParallelAgent(
    name="travel_orchestrator",
    description="Runs hotel, flight, and sightseeing agents in parallel.",
    sub_agents=[hotel_agent, flight_agent, sightseeing_agent],
)

# ── Session & Runner ───────────────────────────────────────────────────────────

session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name="travel_assistant",
    session_service=session_service,
)


# ── Chat helper ────────────────────────────────────────────────────────────────

async def chat(user_id, session_id, user_message):
    """Send one message and print each sub-agent's reply as it arrives."""
    print(f"\nYou : {user_message}")

    message = types.Content(
        role="user",
        parts=[types.Part(text=user_message)]
    )

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        # google_search is a built-in tool — it goes through grounding, not
        # function calls, so we detect its use via grounding_metadata instead
        # of event.get_function_calls().
        if event.grounding_metadata:
            print(f"  [{event.author}] used google_search")

        # Each sub-agent emits its own final response event.
        # is_final_response() returns True for each of the three agents,
        # so we get three labelled responses per user query.
        if event.is_final_response():
            if event.content and event.content.parts:
                text = "".join(p.text for p in event.content.parts if p.text)
                if text:
                    print(f"\n[{event.author}]:\n{text}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    USER_ID    = "traveller_01"
    SESSION_ID = str(uuid.uuid4())

    await session_service.create_session(
        app_name="travel_assistant",
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    print(f"Session ID: {SESSION_ID}")
    print("=" * 60)

    # All three agents answer this query simultaneously.
    # hotel_agent → accommodation options in Japan
    # flight_agent → flight routes and airlines to Japan
    # sightseeing_agent → attractions and activities in Japan
    await chat(USER_ID, SESSION_ID, "I'm planning a 7-day trip to Japan. What should I know?")

    print("\n" + "-" * 60)

    # Second query — each agent answers from its own domain again, in parallel.
    await chat(USER_ID, SESSION_ID, "What's the best time to visit and what can I do there?")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
