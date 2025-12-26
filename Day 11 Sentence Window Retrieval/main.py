import os
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from documents import documents


def build_sentence_window_index(
    documents,
    llm,
    embed_model="BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index",
):
    # parse documents into nodes with surrounding context window
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # load embedding model from huggingface
    embed_model_instance = HuggingFaceEmbedding(
        model_name=embed_model,
        trust_remote_code=True,
        cache_folder="./.cache/huggingface",
    )

    # set global settings for llamaindex
    Settings.llm = llm
    Settings.embed_model = embed_model_instance
    Settings.node_parser = node_parser

    # create new index or load existing one
    if not os.path.exists(save_dir):
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=save_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=save_dir)
        index = load_index_from_storage(storage_context)

    return index


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # swap original sentence with full window context
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    
    # rerank results using cross-encoder for better relevance
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n,
        model="BAAI/bge-reranker-base"
    )

    # build query engine with postprocessing pipeline
    engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank],
    )
    return engine


if __name__ == "__main__":
    # grab api key from environment
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    
    # init llm with openrouter
    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=openrouter_api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
    )

    # build the index with window size 3
    index = build_sentence_window_index(
        documents,
        llm=llm,
        sentence_window_size=3,
        save_dir="sentence_index"
    )

    # create query engine
    query_engine = get_sentence_window_query_engine(index)
    
    # simple chat loop
    while True:
        query = input("\nQuery: ")
        #response
        response = query_engine.query(query)
        print(f"Response: {response}\n")