# HyDE RAG with OpenRouter

from openai import OpenAI
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import os

load_dotenv()


class HyDERAG:
    
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY") 
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.knowledge_base = []
        self.doc_embeddings = []
        self.vocab = []
        
    def add_documents(self, documents: List[str]):
        self.knowledge_base.extend(documents)
        
        # Build fixed vocabulary
        vocab_set = set()
        for doc in self.knowledge_base:
            vocab_set.update(doc.lower().split())
        self.vocab = sorted(vocab_set)
        
        # Compute embeddings
        for doc in documents:
            emb = self._get_embedding(doc)
            self.doc_embeddings.append(emb)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        # Simple bag-of-words embedding
        words = text.lower().split()
        embedding = np.zeros(len(self.vocab))
        
        for i, word in enumerate(self.vocab):
            embedding[i] = words.count(word)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    
    def _generate_hypothetical_doc(self, query: str) -> str:
        # Generate hypothetical answer
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Write a detailed answer to: {query}\n\nProvide a comprehensive response."
            }],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content
    
    def _retrieve_docs(self, hypothetical_doc: str, top_k: int = 3) -> List[Tuple[str, float]]:
        # Find similar docs
        if not self.knowledge_base:
            raise ValueError("No documents in knowledge base")
        
        hypo_emb = self._get_embedding(hypothetical_doc)
        
        similarities = []
        for doc, doc_emb in zip(self.knowledge_base, self.doc_embeddings):
            sim = self._cosine_similarity(hypo_emb, doc_emb)
            similarities.append((doc, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _generate_answer(self, query: str, retrieved_docs: List[Tuple[str, float]]) -> str:
        # Generate final answer
        context = "\n\n".join([f"Document {i+1}:\n{doc}" 
                               for i, (doc, _) in enumerate(retrieved_docs)])
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            }],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    def query(self, question: str, top_k: int = 3) -> str:
        # Full pipeline
        hypo_doc = self._generate_hypothetical_doc(question)
        retrieved = self._retrieve_docs(hypo_doc, top_k)
        answer = self._generate_answer(question, retrieved)
        return answer


# Utils
def load_from_file(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return [doc.strip() for doc in content.split('\n\n') if doc.strip()]


# Test
if __name__ == "__main__":
    rag = HyDERAG()
    
    # Load docs
    docs = [
        "Neural networks consist of layers of interconnected nodes that process information through weighted connections. Training adjusts these weights via backpropagation.",
        "Gradient descent optimizes model parameters by moving in the direction of steepest descent of the loss function.",
        "Transformers use self-attention mechanisms to process sequences in parallel. Powers models like BERT and GPT.",
        "Overfitting happens when models learn training data too well. Prevention: regularization, dropout, cross-validation.",
        "CNNs use convolutional layers with learnable filters to detect hierarchical features in images."
    ]
    
    rag.add_documents(docs)
    
    # Query
    query = input("Question: ").strip()
    
    if query:
        answer = rag.query(query, top_k=3)
        print(f"\n{answer}")