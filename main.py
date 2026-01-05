from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    city = "Lahore"
    summary_template = "tell me weather of {city}"
    summary_prompt_template = PromptTemplate(
        input_variables=["city"],template=summary_template
    )

    llm = ChatOpenAI(temperature=0,model="gpt-5")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"city":city})
    print(response.content)



if __name__ == "__main__":
    main()
