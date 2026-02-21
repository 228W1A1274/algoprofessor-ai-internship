"""
ragas_eval.py
=============
Evaluates the RAG pipeline using the RAGAS framework.

The four core RAGAS metrics:
────────────────────────────────────────────────────────────────
METRIC              MEASURES                    NEEDS GROUND TRUTH?
────────────────────────────────────────────────────────────────
Faithfulness        Is the answer grounded in   No (uses LLM judge)
                    the retrieved context?       

Answer Relevancy    Does the answer actually     No (uses LLM judge)
                    address the question?

Context Precision   Are retrieved docs relevant  Yes (needs ground truth)
                    to the ground truth?

Context Recall      Does retrieved context       Yes (needs ground truth)
                    contain all needed info?
────────────────────────────────────────────────────────────────

RAGAS works by using an LLM-as-judge approach:
It sends the (question, context, answer) triplet to an LLM (usually GPT-4)
and asks structured questions to score each metric from 0 to 1.

Input format (the "evaluation dataset"):
{
    "question": ["What is X?", "How does Y work?"],
    "answer": ["X is ...", "Y works by ..."],
    "contexts": [["chunk1", "chunk2"], ["chunk3"]],  # retrieved chunks (as strings)
    "ground_truth": ["The ground truth answer to X.", "The ground truth answer to Y."],
}

Output: A score from 0 to 1 for each metric, for each question.
Perfect score = 1.0. Acceptable threshold: > 0.7.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# RAGAS core imports
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,           # = Was the answer grounded in context?
    AnswerRelevancy,        # = Did the answer address the question?
    ContextPrecision,       # = Were retrieved chunks relevant?
    ContextRecall,          # = Did retrieved chunks cover the ground truth?
)

# RAGAS uses LangChain LLMs and embeddings under the hood
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Hugging Face datasets library — RAGAS uses this for its data format
from datasets import Dataset

# Import our pipeline
from rag_pipeline import RAGPipeline, RAGConfig
from chunking import SAMPLE_DOCS

load_dotenv()


# =============================================================================
# EVALUATION DATASET
# =============================================================================

# This is our ground truth dataset.
# In production, these would be created by domain experts reviewing documents.
# For this tutorial, we've created them based on SAMPLE_TEXT from chunking.py.
#
# CRITICAL: The quality of your evaluation is only as good as your ground truth.
# Bad ground truth → misleading metrics → wrong optimization decisions.

EVALUATION_DATASET = [
    {
        "question": "What is Retrieval-Augmented Generation (RAG)?",
        "ground_truth": (
            "RAG is a technique that combines large language models with "
            "information retrieval systems, allowing the model to access "
            "external knowledge sources at inference time instead of relying "
            "solely on knowledge encoded in the model's weights during training."
        ),
    },
    {
        "question": "What is the role of reranking in a RAG pipeline?",
        "ground_truth": (
            "Reranking is a post-retrieval step that applies a more "
            "computationally expensive but more accurate relevance model to "
            "re-score initially retrieved candidates. Cohere's reranker is a "
            "cross-encoder that jointly processes the query and each document, "
            "enabling richer relevance assessment than pure vector similarity."
        ),
    },
    {
        "question": "What are the four metrics measured by RAGAS?",
        "ground_truth": (
            "RAGAS measures faithfulness (whether the answer is supported by "
            "the retrieved context), answer relevancy (whether the answer "
            "addresses the question), context precision (whether retrieved "
            "chunks are relevant), and context recall (whether all necessary "
            "information was retrieved)."
        ),
    },
    {
        "question": "What is semantic chunking and how does it work?",
        "ground_truth": (
            "Semantic chunking splits text by meaning using embedding similarity. "
            "It embeds each sentence, computes cosine distances between adjacent "
            "sentences, and identifies breakpoints where distances spike, "
            "indicating topic shifts. Splits occur at these breakpoints so each "
            "chunk contains semantically related sentences."
        ),
    },
    {
        "question": "What is HyDE in the context of RAG query transformation?",
        "ground_truth": (
            "HyDE (Hypothetical Document Embeddings) is a query transformation "
            "technique that generates a hypothetical answer to the query and "
            "uses that hypothetical answer for retrieval, rather than the "
            "original query itself."
        ),
    },
]


# =============================================================================
# METRIC DEFINITIONS WITH EXPLANATIONS
# =============================================================================

def get_metrics():
    """
    Instantiate RAGAS metric objects.
    
    Each metric is a class that internally uses an LLM judge.
    
    Faithfulness:
        Algorithm: Breaks the answer into atomic claims, then for each claim,
        asks the LLM judge "Is this claim supported by the context?"
        Score = (number of supported claims) / (total claims)
        Catch: An answer can be "faithful" to context even if the context is wrong.
        It only checks internal consistency, not factual correctness.
    
    AnswerRelevancy:
        Algorithm: Generates N questions from the answer using the LLM, then
        measures the average cosine similarity between those generated questions
        and the original question using embeddings.
        Score near 1.0 = The answer talks about what the question asked.
        Score near 0.0 = The answer is about something else entirely.
        Catch: Doesn't measure whether the answer is CORRECT, just RELEVANT.
    
    ContextPrecision:
        Algorithm: For each retrieved chunk, asks "Is this chunk relevant to
        the ground truth?" Then computes mean average precision (MAP) which
        rewards systems that put relevant chunks EARLIER in the ranking.
        Score = Weighted mean average precision across all positions.
        Catch: Requires ground truth answers. Tests the RETRIEVER, not the LLM.
    
    ContextRecall:
        Algorithm: Breaks ground truth into atomic claims, then for each claim
        asks "Is this claim covered by the retrieved context?"
        Score = (covered claims) / (total ground truth claims)
        Catch: If your retriever misses a key document, this score will be low
        regardless of how good your LLM is.
    """
    return [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]


# =============================================================================
# DATASET BUILDER
# =============================================================================

def build_evaluation_dataset(
    pipeline: RAGPipeline,
    eval_items: List[Dict[str, str]],
    verbose: bool = True,
) -> Dataset:
    """
    Runs each evaluation question through the RAG pipeline and builds
    the dataset structure RAGAS expects.
    
    RAGAS expects a HuggingFace Dataset with these exact column names:
    - "question": List[str]
    - "answer": List[str]        ← generated by our pipeline
    - "contexts": List[List[str]] ← retrieved chunk texts (not Documents)
    - "ground_truth": List[str]   ← our manually written ground truths
    
    This function is the bridge between our pipeline and RAGAS.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    total = len(eval_items)
    
    for idx, item in enumerate(eval_items, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        if verbose:
            print(f"  [{idx}/{total}] Querying: '{question[:60]}...'")
        
        try:
            # Run the full pipeline for this question
            result = pipeline.query(question)
            
            # RAGAS needs context as List[str] (plain strings, not Document objects)
            # doc.page_content extracts just the text
            context_texts = [
                doc.page_content 
                for doc in result["source_documents"]
            ]
            
            questions.append(question)
            answers.append(result["answer"])
            contexts.append(context_texts)
            ground_truths.append(ground_truth)
            
            if verbose:
                print(f"         ✓ Retrieved {len(context_texts)} docs, "
                      f"answer: '{result['answer'][:80]}...'")
        
        except Exception as e:
            print(f"         ✗ Failed: {e}")
            # Add empty results so the dataset stays aligned
            questions.append(question)
            answers.append("Error generating answer.")
            contexts.append([""])
            ground_truths.append(ground_truth)
    
    # Convert to HuggingFace Dataset format
    # Dataset.from_dict takes a dict of {column_name: [values]}
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    if verbose:
        print(f"\n[Dataset] Built evaluation dataset with {len(dataset)} rows")
    
    return dataset


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_pipeline(
    pipeline: RAGPipeline,
    eval_items: List[Dict[str, str]] = None,
    save_results: bool = True,
    results_path: str = "ragas_results.json",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Runs the complete RAGAS evaluation and returns a dict of metric scores.
    
    Parameters:
        pipeline: A built RAGPipeline instance
        eval_items: List of {question, ground_truth} dicts
        save_results: Whether to save detailed results to JSON
        results_path: Where to save the results
        verbose: Print progress and results
    
    Returns:
        Dict[metric_name → score] e.g. {"faithfulness": 0.87, ...}
    """
    eval_items = eval_items or EVALUATION_DATASET
    
    if verbose:
        print("\n" + "=" * 60)
        print("RAGAS EVALUATION")
        print("=" * 60)
        print(f"Evaluating {len(eval_items)} questions...")
        print("This will make multiple LLM API calls. Please wait.\n")
    
    # Step 1: Build the evaluation dataset
    print("[Step 1/3] Running pipeline on evaluation questions...")
    dataset = build_evaluation_dataset(pipeline, eval_items, verbose=verbose)
    
    # Step 2: Configure RAGAS to use our LLM and embeddings
    # RAGAS needs an LLM for its judge prompts and embeddings for AnswerRelevancy
    print("\n[Step 2/3] Configuring RAGAS LLM judge...")
    judge_llm = ChatOpenAI(
        model="gpt-4o-mini",  # Using mini to reduce costs; gpt-4o is more accurate
        temperature=0,         # Must be 0 for deterministic scoring
    )
    judge_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    metrics = get_metrics()
    
    # Inject our LLM and embeddings into each metric
    # Some metrics need the LLM (Faithfulness, ContextPrecision, ContextRecall)
    # AnswerRelevancy needs both LLM and embeddings
    for metric in metrics:
        metric.llm = judge_llm
        if hasattr(metric, 'embeddings'):
            metric.embeddings = judge_embeddings
    
    # Step 3: Run evaluation
    print("[Step 3/3] Running RAGAS evaluation (LLM judge scoring)...")
    print("          This makes ~4-8 LLM calls per question. Patience!\n")
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,  # Don't crash on individual metric failures
    )
    
    # Convert to dict of {metric_name: avg_score}
    scores = result.to_pandas().mean().to_dict()
    
    # Clean up column names (RAGAS sometimes adds suffixes)
    clean_scores = {}
    metric_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
    }
    for key, value in scores.items():
        for expected_key in metric_map.keys():
            if expected_key in key:
                clean_scores[metric_map[expected_key]] = round(float(value), 4)
    
    # Step 4: Display results
    if verbose:
        _print_evaluation_report(clean_scores, dataset)
    
    # Step 5: Save results
    if save_results:
        _save_results(clean_scores, dataset, results_path)
    
    return clean_scores


# =============================================================================
# REPORTING UTILITIES
# =============================================================================

def _print_evaluation_report(scores: Dict[str, float], dataset: Dataset) -> None:
    """Pretty-print the evaluation results with interpretation guidance."""
    
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION REPORT")
    print("=" * 60)
    print(f"Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sample size: {len(dataset)} questions\n")
    
    # Score interpretation thresholds
    thresholds = {
        "excellent": 0.85,
        "good": 0.70,
        "acceptable": 0.55,
    }
    
    print(f"{'Metric':<25} {'Score':>8} {'Rating':>15} {'Action'}")
    print("-" * 70)
    
    metric_descriptions = {
        "faithfulness": "Answer grounded in context?",
        "answer_relevancy": "Answer relevant to question?",
        "context_precision": "Retrieved chunks relevant?",
        "context_recall": "All needed info retrieved?",
    }
    
    for metric, score in scores.items():
        if score >= thresholds["excellent"]:
            rating = "🟢 Excellent"
            action = "No action needed"
        elif score >= thresholds["good"]:
            rating = "🟡 Good"
            action = "Minor tuning"
        elif score >= thresholds["acceptable"]:
            rating = "🟠 Acceptable"
            action = "Needs improvement"
        else:
            rating = "🔴 Poor"
            action = "Significant rework"
        
        desc = metric_descriptions.get(metric, metric)
        print(f"{desc:<25} {score:>8.4f} {rating:>15}  → {action}")
    
    avg_score = sum(scores.values()) / len(scores) if scores else 0
    print("-" * 70)
    print(f"{'Overall Average':<25} {avg_score:>8.4f}")
    
    print("\n--- Interpretation Guide ---")
    print("  Faithfulness < 0.7  → LLM is hallucinating beyond the context")
    print("                          Fix: Tighten your system prompt grounding instruction")
    print("  Answer Relevancy < 0.7 → Answers are off-topic")
    print("                          Fix: Improve your prompt's answer format instructions")
    print("  Context Precision < 0.7 → Retrieving irrelevant documents")
    print("                          Fix: Better chunking, stricter reranker top_n")
    print("  Context Recall < 0.7 → Missing key information during retrieval")
    print("                          Fix: Increase retrieval K, improve embedding model")
    print("=" * 60)


def _save_results(
    scores: Dict[str, float],
    dataset: Dataset,
    path: str,
) -> None:
    """Save full evaluation results to JSON for later analysis."""
    
    # Convert dataset to list of dicts for JSON serialization
    dataset_records = []
    for i in range(len(dataset)):
        record = {
            "question": dataset[i]["question"],
            "answer": dataset[i]["answer"],
            "ground_truth": dataset[i]["ground_truth"],
            "context_count": len(dataset[i]["contexts"]),
            "contexts": dataset[i]["contexts"],
        }
        dataset_records.append(record)
    
    output = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "aggregate_scores": scores,
        "sample_count": len(dataset),
        "individual_results": dataset_records,
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Saved] Full results saved to '{path}'")


# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================

def evaluate_single_question(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Dict[str, float]:
    """
    Evaluate a single question-answer pair.
    Useful for debugging specific failures in your pipeline.
    
    Usage:
        scores = evaluate_single_question(
            question="What is RAG?",
            answer="RAG is a technique that...",
            contexts=["Retrieval-Augmented Generation combines..."],
            ground_truth="RAG combines LLMs with retrieval systems..."
        )
    """
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    })
    
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    metrics = get_metrics()
    for metric in metrics:
        metric.llm = judge_llm
        if hasattr(metric, 'embeddings'):
            metric.embeddings = judge_embeddings
    
    result = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False)
    scores = result.to_pandas().iloc[0].to_dict()
    
    print(f"\nSingle Question Evaluation:")
    print(f"  Q: {question}")
    print(f"  A: {answer[:100]}...")
    for k, v in scores.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    
    return scores


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing ragas_eval.py")
    print("=" * 60)
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not found. Add it to your .env file.")
        exit(1)
    if not os.getenv("COHERE_API_KEY"):
        print("[ERROR] COHERE_API_KEY not found. Add it to your .env file.")
        exit(1)
    
    # Build the RAG pipeline using our sample corpus
    print("\n[Setup] Building RAG pipeline for evaluation...")
    pipeline = RAGPipeline()
    pipeline.build_from_documents(SAMPLE_DOCS)
    
    # Run the evaluation
    # Using only the first 3 questions for speed; remove the slice for full eval
    scores = evaluate_pipeline(
        pipeline=pipeline,
        eval_items=EVALUATION_DATASET[:3],  # Slice for testing; remove for full eval
        save_results=True,
        results_path="ragas_results.json",
        verbose=True,
    )
    
    print("\n[Final Scores]")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")
    
    print("\n[SUCCESS] ragas_eval.py completed successfully!")
    print("Check 'ragas_results.json' for the full detailed report.")