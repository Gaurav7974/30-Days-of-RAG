# Multi-query RAG with OpenRouter

from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

load_dotenv()

# Get API key
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

# Set for LangChain compatibility
os.environ["OPENAI_API_KEY"] = openrouter_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openrouter_key,
    openai_api_base="https://openrouter.ai/api/v1",
)

# Vector DB
vectordb = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Schema for multi-query output
class MultiQueries(BaseModel):
    queries: List[str]

parser = PydanticOutputParser(pydantic_object=MultiQueries)

# Prompt
multiquery_prompt = ChatPromptTemplate.from_messages([
    ("system", "You rewrite search queries for a RAG system."),
    ("user", """Generate 3 different variations of this query to retrieve relevant documents.

Return as JSON:
{format_instructions}

Original query: "{original_query}"
""")
]).partial(format_instructions=parser.get_format_instructions())

# LLM - Fixed model name
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # Correct OpenRouter format
    temperature=0.4,
    openai_api_key=openrouter_key,
    openai_api_base="https://openrouter.ai/api/v1",
)

multiquery_chain = multiquery_prompt | llm | parser


def generate_query_variations(original_query: str) -> List[str]:
    # Generate 3 alternative queries
    mq: MultiQueries = multiquery_chain.invoke({"original_query": original_query})
    return mq.queries


def multi_query_retrieve(original_query: str, k: int = 5):
    # Multi-query retrieval with deduplication
    variations = generate_query_variations(original_query)

    print("Original query:", original_query)
    print("\nGenerated variations:")
    for i, q in enumerate(variations, start=1):
        print(f"{i}. {q}")

    # Retrieve docs for each variation
    all_results = []
    for i, q in enumerate(variations, start=1):
        docs = retriever.invoke(q)  # Use invoke instead
        print(f"\nQuery {i} results: {q}")
        for j, d in enumerate(docs, start=1):
            print(f"  [{j}] {d.page_content[:80]}...")
        all_results.extend(docs)

    # Deduplicate
    seen_ids = set()
    merged = []
    for d in all_results:
        doc_id = d.metadata.get("id") or d.metadata.get("source") or hash(d.page_content)
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged.append(d)

    return merged


if __name__ == "__main__":
    query = "How does Tesla make money?"
    final_docs = multi_query_retrieve(query, k=5)
