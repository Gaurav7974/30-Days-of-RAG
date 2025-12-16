METADATA_FILTER_PROMPT = """
You extract structured metadata from user queries.

Rules:
- Return ONLY a valid JSON object.
- Do NOT include explanations, markdown, or text outside JSON.
- Use only the provided metadata field names.
- Omit fields that are not explicitly mentioned.
- Preserve numeric values as numbers.

Examples:

Query: "What was the revenue of Nvidia in 2022?"
Metadata fields: ["company", "year"]
Output:
{"company": "nvidia", "year": 2022}

Query: "Papers about Alzheimer published in 2023"
Metadata fields: ["disease", "year"]
Output:
{"disease": "Alzheimer", "year": 2023}

Task:

Query: "{{query}}"
Metadata fields: {{metadata_fields}}

Output JSON:
"""
