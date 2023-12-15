import os

from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from getpass import getpass

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/winner/{year}")
def ask_winner(year: int):
    HUGGINGFACEHUB_API_TOKEN = getpass()
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

    template = """Question: {question}

    # Answer: """
    # prompt = PromptTemplate(
    #         template=template,
    #     input_variables=['question']
    # )

    question = f"Who won the FIFA World Cup in the year {year}?"

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    repo_id = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_length": 64}
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    return { "year": llm_chain.run(question) }


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# @app.get("/articles/{article_id}")
# def read_article(article_id: int, q: Union[str, None] = None):
#     return {"article_id": article_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item": item, "item_name": item.name, "item_id": item_id}
