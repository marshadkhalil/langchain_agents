from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
def main():
    load_dotenv()
    city = "Lahore"
    summary_template = "tell me weather of {city}"
    summary_prompt_template = PromptTemplate(
        input_variables=["city"],template=summary_template
    )

    # llm = ChatOpenAI(temperature=0,model="gpt-5")
    # llm = ChatOllama(temperature=0,model="gemma3:270m")
    llm = ChatOllama(temperature=0,model="deepseek-r1:8b")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"city":city})
    print(response.content)