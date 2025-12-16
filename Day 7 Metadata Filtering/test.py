from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

from metadata_extrct import QueryMetadataExtractor


documents = [
    Document(
        content="Nvidia revenue reached $60B in 2022.",
        meta={"company": "nvidia", "year": 2022},
    ),
    Document(
        content="Nvidia revenue was $26B in 2020.",
        meta={"company": "nvidia", "year": 2020},
    ),
    Document(
        content="AMD revenue in 2022 was $23B.",
        meta={"company": "amd", "year": 2022},
    ),
]

document_store = InMemoryDocumentStore()
document_store.write_documents(documents)

metadata_extractor = QueryMetadataExtractor()
retriever = InMemoryBM25Retriever(document_store=document_store)

pipeline = Pipeline()
pipeline.add_component("metadata_extractor", metadata_extractor)
pipeline.add_component("retriever", retriever)

pipeline.connect("metadata_extractor.filters", "retriever.filters")

result = pipeline.run(
    data={
        "metadata_extractor": {
            "query": "What was Nvidia revenue in 2022?",
            "metadata_fields": ["company", "year"],
        },
        "retriever": {"query": "revenue"},
    }
)

print(result["documents"])
