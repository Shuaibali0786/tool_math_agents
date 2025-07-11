import os
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(" GEMINI_API_KEY is missing from .env file.")

@function_tool
def add(a: int, b: int):
    """Add two numbers and return the result"""
    return a + b

agent = Agent(
    name="Math Agent",
    instructions="You are a math assistant. Use the 'add' tool when a user asks to add two numbers.",
    tools=[add]
)


client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client
)

questions = [
    "What is 5 + 7?",
    "Can you add 10 and 15?",
    "Add 100 and 250 please."
]

for q in questions:
    print(f"\n🔸 Question: {q}")
    response = Runner.run_sync(agent, q, run_config=config)
    print("Answer:", response.final_output)
