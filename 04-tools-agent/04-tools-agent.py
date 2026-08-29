import asyncio
import uuid
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# google_search is a built-in ADK tool that lets the agent search the web.
# It uses Google Search under the hood — no extra API key needed with Gemini.
from google.adk.tools import google_search

# BuiltInCodeExecutor lets the agent write and run Python code during a response.
# It uses Gemini's native code execution capability — runs in a sandboxed environment.
from google.adk.code_executors import BuiltInCodeExecutor

load_dotenv()

# ── Agent ──────────────────────────────────────────────────────────────────────

root_agent = LlmAgent(
    name="travel_assistant",
    model="gemini-2.5-flash",

    instruction="""You are a helpful travel assistant with two powerful capabilities:
1. Search the web for up-to-date travel information such as flights, hotels, visa rules, and destinations.
2. Write and execute Python code to perform travel-related calculations like budget breakdowns, currency conversions, and trip cost estimates.

Use search when the user asks about current travel deals, destination info, or real-world facts.
Use code execution when the user asks you to calculate costs, convert currencies, or break down a travel budget.""",

    # tools is a list — you can add as many tools as needed.
    # google_search is a pre-built instance, so we pass it directly (no parentheses).
    # The agent automatically decides when to call a tool based on the user's message.
    tools=[google_search],

    # code_executor is a separate parameter (not part of tools list).
    # BuiltInCodeExecutor uses Gemini's sandboxed code execution environment —
    # the agent can write Python, run it, and include the output in its response.
    code_executor=BuiltInCodeExecutor(),
)

# ── Session & Runner (same pattern as file 02/03) ──────────────────────────────

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
        # ── Detect which tools were used ──────────────────────────────────────
        # google_search is a grounding tool — Gemini signals its use via
        # grounding_metadata on the event, not as a function call.
        if event.grounding_metadata:
            print("  [tool used] google_search")

        # code_executor shows up as an executable_code part in the content,
        # not as a function call either — it's native to the Gemini model.
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.executable_code:
                    print(f"  [tool used] code_execution")

        # ── Print final response ──────────────────────────────────────────────
        if event.is_final_response():
            if event.content and event.content.parts:
                text = "".join(
                    part.text for part in event.content.parts if part.text
                )
                if text:
                    print(f"Agent: {text}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    USER_ID    = "user_01"
    SESSION_ID = str(uuid.uuid4())

    await session_service.create_session(
        app_name="travel_assistant",
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    print(f"Session ID: {SESSION_ID}")
    print("=" * 60)

    # Triggers google_search — agent fetches live flight/hotel info
    await chat(USER_ID, SESSION_ID, "What are the best flight options from New York to Tokyo right now?")

    # Triggers code_executor — agent writes and runs Python to break down the budget
    await chat(USER_ID, SESSION_ID, "I have a $3000 budget for a 7-day trip. Break it down across flights, hotel, food, and activities.")

    # May trigger both — searches current exchange rate then calculates
    await chat(USER_ID, SESSION_ID, "What is today's USD to JPY exchange rate? How much is $3000 in Japanese Yen?")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
