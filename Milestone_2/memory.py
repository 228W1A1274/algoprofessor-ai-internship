"""
memory.py - Conversational memory with sliding window and LLM compression
Enterprise Knowledge Navigator - AlgoProfessor Internship 2026

Features:
- 10-turn sliding window: keeps last 10 Q&A pairs in context
- Summary compression: when window fills, compress oldest 5 turns into a summary
- Eval logging: stores question/answer/context triples for RAGAS evaluation
"""


class ConversationMemory:
    """
    Manages conversation history with summary compression.
    
    When history reaches max_turns, oldest 5 turns are summarized
    into a compressed string. This keeps context size manageable
    without losing all previous information.
    """

    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.history = []       # list of {question, answer} dicts
        self.summary = ""       # compressed summary of older turns
        self.eval_log = []      # for RAGAS evaluation

    def add_turn(self, question, answer, contexts=None):
        """Add a completed Q&A turn to memory."""
        self.history.append({"question": question, "answer": answer})
        if contexts:
            self.eval_log.append({
                "question": question,
                "answer": answer,
                "contexts": contexts
            })
        if len(self.history) >= self.max_turns:
            self._compress()

    def _compress(self, groq_client):
        """Compress oldest 5 turns into a summary paragraph."""
        old = self.history[:5]
        self.history = self.history[5:]
        turns_text = ""
        for t in old:
            turns_text += "Q: " + t["question"] + "\nA: " + t["answer"] + "\n\n"
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": "Summarize these conversation turns in 3 sentences keeping key facts:\n\n" + turns_text
            }],
            max_tokens=150,
            temperature=0.1
        )
        new_summary = response.choices[0].message.content
        self.summary = (self.summary + " " + new_summary).strip()

    def build_context_string(self):
        """Build the conversation history string for inclusion in prompts."""
        parts = []
        if self.summary:
            parts.append("Previous conversation summary: " + self.summary)
        for t in self.history[-5:]:
            parts.append("User: " + t["question"])
            parts.append("Assistant: " + t["answer"])
        return "\n".join(parts)

    def reset(self):
        """Clear all history."""
        self.history = []
        self.summary = ""
        self.eval_log = []

    @property
    def turn_count(self):
        return len(self.history)

    @property
    def is_full(self):
        return len(self.history) >= self.max_turns
