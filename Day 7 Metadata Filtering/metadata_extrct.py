import json
import re
from typing import Dict, List
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator

from metadata_filters import METADATA_FILTER_PROMPT


def safe_json_extract(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


@component()
class QueryMetadataExtractor:

    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.pipeline = Pipeline()

        # explicitly declare required variables
        self.pipeline.add_component(
            "builder",
            PromptBuilder(
                METADATA_FILTER_PROMPT,
                required_variables=["query", "metadata_fields"],
            ),
        )

        # generation params go into generation_kwargs
        self.pipeline.add_component("llm",
            HuggingFaceLocalGenerator(
                model=model_name,
                generation_kwargs={"max_new_tokens": 64},
            ),
        )
        self.pipeline.connect("builder", "llm")

    @component.output_types(filters=Dict)
    def run(self, query: str, metadata_fields: List[str]):

        result = self.pipeline.run(
            {
                "builder": {
                    "query": query,
                    "metadata_fields": metadata_fields,
                }
            }
        )

        raw = result["llm"]["replies"][0]
        extracted = safe_json_extract(raw)

        allowed = set(metadata_fields)
        extracted = {k: v for k, v in extracted.items() if k in allowed}

        if not extracted:
            return {"filters": None}

        conditions = [
            {
                "field": f"meta.{k}",
                "operator": "==",
                "value": v,
            }
            for k, v in extracted.items()
        ]

        return {
            "filters": {
                "operator": "AND",
                "conditions": conditions,
            }
        }
