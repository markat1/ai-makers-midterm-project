import chainlit as cl
from chainlit.playground.providers import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain.prompts import ChatPromptTemplate
from operator import  itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.retrievers import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

template = """
you can only answer questions related to what's in the context. If it's not in the context, then you would reply with 
'Sorry I have no answer to your particular question. I can only answer things regarding: {context}'

Context:
{context}

Question:
{question}
"""

init_settings = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


load_dotenv()

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        text,
    )
    return len(tokens)


@cl.on_chat_start
async def main():
    model = ChatOpenAI(streaming=True)
    
    prompt = ChatPromptTemplate.from_template(template)

    nvida_doc = PyMuPDFLoader('../docs/nvidia-document.pdf')
    data = nvida_doc.load()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1700,
    chunk_overlap = 0,
    length_function = tiktoken_len)
    
    nvidia_doc_chunks = text_splitter.split_documents(data)

    vector_store = FAISS.from_documents(nvidia_doc_chunks, embedding=embeddings)
    
    retriever = vector_store.as_retriever()
    advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=model)

    runnable = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | model, "context": itemgetter("context")})

    # retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # document_chain = create_stuff_documents_chain(model, retrieval_qa_prompt)
    # runnable = create_retrieval_chain(advanced_retriever, document_chain)

    # cl.user_session.set("settings", init_settings)
    # cl.user_session.set("nvidia_doc", data)

    cl.user_session.set("runnable", runnable)



@cl.on_message
async def on_message(message: cl.Message):
    # settings = cl.user_session.get("settings")
    # nvida_doc = cl.user_session.get("nvidia_doc")
    runnable = cl.user_session.get("runnable")

    msg = cl.Message(content="")

    
    # async for chunk in runnable.astream(
    #     {"question": message.content},
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk, True)

    # await msg.send()

    inputs = {"question": message.content}
    result = await runnable.ainvoke(inputs)
    msg = cl.Message(content=result["response"].content)
    await msg.send()


 




