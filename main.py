from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
print("API KEY loaded:", bool(os.getenv("GOOGLE_API_KEY")))

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

# ── Pydantic output schema ──────────────────────────────────────────────────
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# ── At least one real tool ──────────────────────────────────────────────────
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wiki]

# ── Prompt (ReAct format requires these exact placeholders) ─────────────────
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a research assistant. Use tools to answer the question.
After gathering information, return ONLY a JSON object matching this schema:
{format_instructions}

Use this ReAct format:
Thought: ...
Action: tool_name
Action Input: your input
Observation: ...
Thought: I now know the answer
Final Answer: <your JSON here>
""",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# ── Modern agent setup ──────────────────────────────────────────────────────
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# ── Run ─────────────────────────────────────────────────────────────────────
raw_response = agent_executor.invoke({"input": "What is the capital of France?"})

# Parse the output into the Pydantic model
try:
    structured = parser.parse(raw_response["output"])
    print(structured)
except Exception as e:
    print("Raw output:", raw_response["output"])
    print("Parse error:", e)