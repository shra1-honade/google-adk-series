import asyncio
import uuid
import requests
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search

load_dotenv()

# ── Custom Tool ────────────────────────────────────────────────────────────────
#
# Verbatim from 05-custom-tool-agent.py.
# Used only by weather_agent — shows that a parallel group can mix custom tools
# and built-in tools (google_search) without any special wiring.

def get_weather(latitude: float, longitude: float) -> dict:
    """Get the current weather for a travel destination using its coordinates.

    Use this tool when the user asks about weather at a destination.
    Returns current temperature, wind speed, and weather condition code.

    Args:
        latitude: The latitude of the destination (e.g. 35.6762 for Tokyo).
        longitude: The longitude of the destination (e.g. 139.6503 for Tokyo).

    Returns:
        A dictionary containing current weather data for the destination.
    """
    # Open-Meteo is a free weather API — no API key required.
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True,   # returns temperature, wind speed, weather code
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


# ── Parallel sub-agents (Step 1 of the pipeline) ───────────────────────────────
#
# Each agent has output_key set — ADK persists the agent's final text response
# into session.state[output_key] via event.actions.state_delta.  The downstream
# travel_planner_agent reads those values via {key} placeholders in its instruction.

hotel_agent = LlmAgent(
    name="hotel_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a hotel specialist. "
        "Use google_search to find current hotel options, prices, and neighbourhoods. "
        "Only answer questions about accommodation."
    ),
    tools=[google_search],
    output_key="hotel_findings",   # → session.state["hotel_findings"]
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
    output_key="flight_findings",  # → session.state["flight_findings"]
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
    output_key="sightseeing_findings",  # → session.state["sightseeing_findings"]
)

# weather_agent uses a custom tool — get_weather — while the three above use
# google_search (a built-in tool).  Both tool types work inside the same
# ParallelAgent without any special configuration.
weather_agent = LlmAgent(
    name="weather_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a weather specialist. "
        "When given a destination, derive its coordinates and call get_weather "
        "to get the current weather. "
        "Report temperature, conditions, and packing advice."
    ),
    tools=[get_weather],
    output_key="weather_findings",  # → session.state["weather_findings"]
)

# ── Step 1: ParallelAgent ──────────────────────────────────────────────────────
#
# All four sub-agents receive the same user query simultaneously and run
# concurrently via asyncio.TaskGroup.  Each runs in its own isolated branch
# context so their conversation histories don't interfere.
# Responses arrive in non-deterministic order — whichever finishes first.

travel_researcher = ParallelAgent(
    name="travel_researcher",
    description="Gathers hotel, flight, sightseeing, and weather info in parallel.",
    sub_agents=[hotel_agent, flight_agent, sightseeing_agent, weather_agent],
)

# ── Step 2: Coordinator / summariser (no tools) ────────────────────────────────
#
# This agent has no tools — it is a pure coordinator that reads the four
# findings already stored in session.state by the parallel step.
#
# ADK substitutes {key} placeholders in the instruction string at runtime
# using the current session.state values, so the planner receives all four
# research blobs in its system prompt without any extra wiring.

travel_planner_agent = LlmAgent(
    name="travel_planner_agent",
    model="gemini-2.5-flash",
    instruction="""You are a senior travel planner. Your researchers have gathered all the information you need. Synthesise it into one clear, friendly travel plan.

HOTEL FINDINGS:
{hotel_findings}

FLIGHT FINDINGS:
{flight_findings}

SIGHTSEEING FINDINGS:
{sightseeing_findings}

WEATHER FINDINGS:
{weather_findings}

Write a concise travel plan covering:
1. Recommended flights and booking tips
2. Best hotel areas and options
3. Top things to do and see
4. Current weather and what to pack
End with a "Quick Tips" section.""",
    # No tools — the planner only synthesises pre-gathered facts.
)

# ── SequentialAgent (root) ─────────────────────────────────────────────────────
#
# SequentialAgent guarantees that Step 1 (travel_researcher / parallel research)
# completes fully before Step 2 (travel_planner_agent / synthesis) begins.
# This creates a three-level nesting:
#   SequentialAgent → ParallelAgent → LlmAgent
#
# The guarantee means the planner always has all four output_key values in
# session.state by the time its instruction string is rendered.

root_agent = SequentialAgent(
    name="travel_pipeline",
    description="Research phase (parallel) then planning phase (sequential synthesis).",
    sub_agents=[travel_researcher, travel_planner_agent],
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
    """Send one message and stream events from the full pipeline."""
    print(f"\nYou : {user_message}")

    message = types.Content(
        role="user",
        parts=[types.Part(text=user_message)],
    )

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        # google_search (built-in) is detected via grounding_metadata.
        if event.grounding_metadata:
            print(f"  [{event.author}] used google_search")

        # get_weather (custom tool) goes through ADK's function call system,
        # so it appears in get_function_calls() — unlike built-in tools.
        for fn in event.get_function_calls():
            print(f"  [{event.author}] called {fn.name}({fn.args})")

        if event.is_final_response():
            if event.content and event.content.parts:
                text = "".join(p.text for p in event.content.parts if p.text)
                if text:
                    # event.author distinguishes the planner's synthesis from
                    # the individual researcher responses.
                    if event.author == "travel_planner_agent":
                        print(f"\n{'='*60}\n[TRAVEL PLAN]\n{'='*60}\n{text}")
                    else:
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

    # Pipeline flow:
    # 1. travel_researcher (ParallelAgent) fans out to all four sub-agents at once.
    #    Each stores its result in session.state via output_key.
    # 2. travel_planner_agent reads those four values from session.state via
    #    {key} placeholders and synthesises a unified travel plan.
    await chat(
        USER_ID,
        SESSION_ID,
        "I'm planning a 7-day trip to Tokyo, Japan. Give me a full travel plan.",
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
