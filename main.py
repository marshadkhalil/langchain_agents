from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()
"""
initialize tavily sdk
"""
# tavily = TavilyClient()
"""
this is custom tool which using tavily sdk for search but 
langchain has its own package for internet searching
"""
# @tool
# def search(query:str)->str:
#     """
#     Tool that searches over internet
#     Args:
#         query:The query to search for
#     Returns:
#         The search result
#     """
#     return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5")
# custom search tool
# tools = [search]
# tavily search tool
tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools)

def main():
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job positions for an ai engineer using langchain in the bay area on linkedin and list their description")})
    print(result)



if __name__ == "__main__":
    main()
