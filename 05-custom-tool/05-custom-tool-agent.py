import asyncio
import uuid
import requests
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

load_dotenv()

# ── Custom Tool ────────────────────────────────────────────────────────────────

# In ADK, any plain Python function can become a tool.
# The agent uses the function name, parameters, and docstring to understand
# when and how to call it — so clear naming and docstrings are important.
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
        resp.raise_for_status()   # raises an error for 4xx/5xx responses
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


# ── Agent ──────────────────────────────────────────────────────────────────────

root_agent = LlmAgent(
    name="travel_assistant",
    model="gemini-2.5-flash",

    instruction="""You are a helpful travel assistant.
When a user asks about weather at a destination, use the get_weather tool.
Provide the coordinates yourself based on the destination name.
Interpret the weather code and give a friendly, human-readable weather summary.""",

    # Pass the function directly — ADK wraps it into a FunctionTool automatically.
    # The agent reads the docstring to know when and how to call it.
    tools=[get_weather],
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
        # get_function_calls() works for custom tools — unlike built-in tools
        # (google_search, code_executor), custom tools go through ADK's
        # function call system, so they show up here.
        for fn in event.get_function_calls():
            print(f"  [tool called] {fn.name}({fn.args})")

        if event.is_final_response():
            if event.content and event.content.parts:
                text = "".join(
                    part.text for part in event.content.parts if part.text
                )
                if text:
                    print(f"Agent: {text}")


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

    # Agent will call get_weather with Tokyo's coordinates
    await chat(USER_ID, SESSION_ID, "What's the weather like in Tokyo right now?")

    # Agent uses session memory — knows we're still talking about Tokyo
    await chat(USER_ID, SESSION_ID, "Is it a good time to visit given this weather?")

    # Agent calls get_weather again with different coordinates
    await chat(USER_ID, SESSION_ID, "How about the weather in Paris?")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
