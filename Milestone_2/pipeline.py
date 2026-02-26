"""
pipeline.py - Full RAG pipeline connecting all components
Enterprise Knowledge Navigator - AlgoProfessor Internship 2026

Components:
- KnowledgeGraph: Neo4j graph with local fallback
- RAGPipeline: full end-to-end pipeline (retrieve -> rerank -> generate)
- ConversationalRAG: stateful pipeline with 10-turn memory
"""

import json
import time


class KnowledgeGraph:
    """
    Neo4j knowledge graph with automatic local dictionary fallback.
    Stores entities (nodes) and relationships (edges) extracted from documents.
    Enables graph-enhanced retrieval by expanding queries with related entities.
    """

    def __init__(self, uri=None, user=None, password=None):
        self.use_neo4j = False
        self.driver = None
        self.local_graph = {"entities": {}, "relationships": []}
        if uri and user and password:
            try:
                from neo4j import GraphDatabase
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                self.driver.verify_connectivity()
                self._setup_constraints()
                self.use_neo4j = True
                print("KnowledgeGraph: connected to Neo4j AuraDB")
            except Exception as e:
                print("KnowledgeGraph: Neo4j failed (" + str(e)[:60] + "), using local fallback")
        else:
            print("KnowledgeGraph: using local dictionary fallback")

    def _setup_constraints(self):
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            except Exception:
                pass

    def add(self, extracted, source):
        """Add entities and relationships from an extraction result dict."""
        entities = extracted.get("entities", [])
        relationships = extracted.get("relationships", [])
        if self.use_neo4j and self.driver:
            with self.driver.session() as session:
                for ent in entities:
                    if ent.get("name") and ent.get("type"):
                        session.run(
                            "MERGE (e:Entity {name: $name}) SET e.type = $type, e.source = $source",
                            name=ent["name"], type=ent["type"], source=source
                        )
                for rel in relationships:
                    if rel.get("from") and rel.get("to") and rel.get("relation"):
                        session.run(
                            "MERGE (a:Entity {name: $fn}) "
                            "MERGE (b:Entity {name: $tn}) "
                            "MERGE (a)-[r:RELATES {type: $rel}]->(b) "
                            "SET r.source = $source",
                            fn=rel["from"], tn=rel["to"],
                            rel=rel["relation"], source=source
                        )
        else:
            for ent in entities:
                if ent.get("name"):
                    self.local_graph["entities"][ent["name"]] = ent.get("type", "UNKNOWN")
            for rel in relationships:
                self.local_graph["relationships"].append({**rel, "source": source})

    def find_related(self, entity_names):
        """Return entity names related to given entities via graph edges."""
        related = set()
        if self.use_neo4j and self.driver:
            with self.driver.session() as session:
                for name in entity_names:
                    result = session.run(
                        "MATCH (e:Entity {name: $name})-[r]->(related) "
                        "RETURN related.name as name LIMIT 10",
                        name=name
                    )
                    for record in result:
                        related.add(record["name"])
        else:
            for rel in self.local_graph["relationships"]:
                if rel.get("from") in entity_names:
                    related.add(rel.get("to", ""))
                if rel.get("to") in entity_names:
                    related.add(rel.get("from", ""))
        return list(related - set(entity_names))

    def stats(self):
        if self.use_neo4j and self.driver:
            with self.driver.session() as session:
                n = session.run("MATCH (e:Entity) RETURN count(e) as c").single()["c"]
                r = session.run("MATCH ()-[r:RELATES]->() RETURN count(r) as c").single()["c"]
                return {"nodes": n, "relationships": r, "backend": "Neo4j AuraDB"}
        return {
            "nodes": len(self.local_graph["entities"]),
            "relationships": len(self.local_graph["relationships"]),
            "backend": "local fallback"
        }

    def close(self):
        if self.driver:
            self.driver.close()


class RAGPipeline:
    """
    Full end-to-end RAG pipeline.
    Takes a question, retrieves relevant chunks, generates a cited answer.
    """

    def __init__(self, retriever, groq_client, kg=None):
        self.retriever = retriever
        self.groq_client = groq_client
        self.kg = kg

    def _call_groq(self, prompt, max_tokens=400):
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful Tesla company expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content

    def _build_context(self, chunks):
        """Build numbered context string with source labels."""
        parts = []
        sources = []
        for i, chunk in enumerate(chunks):
            label = "[Source " + str(i+1) + ": " + chunk["metadata"]["source"] + " page " + str(chunk["metadata"]["page_number"]) + "]"
            parts.append(label + "\n" + chunk["text"])
            sources.append({
                "label": label,
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page_number"],
                "category": chunk["metadata"].get("category", "general"),
                "rerank_score": chunk.get("rerank_score", 0)
            })
        return "\n\n".join(parts), sources

    def run(self, query, history_context=""):
        """
        Run full RAG pipeline for one query.
        Returns dict with answer, sources, and retrieved chunks.
        """
        # Retrieve top chunks
        top_chunks = self.retriever.retrieve(query, use_hyde=True, use_multiquery=True, top_n=5)

        # Build context
        context, sources = self._build_context(top_chunks)

        # Build prompt
        prompt = "Answer the question using ONLY the context below.\n"
        prompt += "Cite sources like [Source 1] when using information.\n"
        prompt += "If context lacks the answer, say so clearly.\n\n"
        if history_context:
            prompt += "Conversation history:\n" + history_context + "\n\n"
        prompt += "Context:\n" + context + "\n\n"
        prompt += "Question: " + query + "\n\nAnswer:"

        answer = self._call_groq(prompt, max_tokens=400)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "chunks": top_chunks
        }


class ConversationalRAG:
    """
    Stateful conversational RAG with 10-turn memory and summary compression.
    Wraps RAGPipeline with ConversationMemory.
    """

    def __init__(self, retriever, groq_client, kg=None, max_turns=10):
        self.pipeline = RAGPipeline(retriever, groq_client, kg)
        self.groq_client = groq_client
        self.max_turns = max_turns
        self.history = []
        self.summary = ""
        self.eval_log = []

    def _compress(self):
        """Compress oldest 5 turns into summary."""
        old = self.history[:5]
        self.history = self.history[5:]
        turns_text = "".join(["Q: " + t["question"] + "\nA: " + t["answer"] + "\n\n" for t in old])
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Summarize in 3 sentences keeping key facts:\n\n" + turns_text}],
            max_tokens=150,
            temperature=0.1
        )
        self.summary = (self.summary + " " + response.choices[0].message.content).strip()

    def _history_context(self):
        parts = []
        if self.summary:
            parts.append("Summary: " + self.summary)
        for t in self.history[-5:]:
            parts.append("User: " + t["question"])
            parts.append("Assistant: " + t["answer"])
        return "\n".join(parts)

    def chat(self, question):
        """Process one conversation turn."""
        if len(self.history) >= self.max_turns:
            self._compress()

        result = self.pipeline.run(question, history_context=self._history_context())

        self.history.append({"question": question, "answer": result["answer"]})
        self.eval_log.append({
            "question": question,
            "answer": result["answer"],
            "contexts": [c["text"] for c in result["chunks"]]
        })

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "turn": len(self.history),
            "contexts": [c["text"] for c in result["chunks"]]
        }

    def reset(self):
        self.history = []
        self.summary = ""
        self.eval_log = []
