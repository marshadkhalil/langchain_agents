from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_classic.prompts import PromptTemplate

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
# it will not work for langsmith endpoint with eu
# react_prompt = hub.pull("hwchase17/react")
# below is custom react prompt
template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
react_prompt = PromptTemplate(
    input_variables=[
        "input",
        "tools",
        "tool_names",
        "agent_scratchpad",
    ],template=template)
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)
agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)
chain = agent_executor

def main():
    result = chain.invoke(
        input={
            "input":"search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
        }
    )
    print(result)

if __name__ == "__main__":
    main()
