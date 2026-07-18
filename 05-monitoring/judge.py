import json
import os

from pydantic import BaseModel
from typing import Literal
from groq import Groq
from dotenv import load_dotenv

from evaluation_utils import llm_structured_retry

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str

judge_instructions = """
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer addresses the question
- PARTLY_RELEVANT: the answer partially addresses the question
- NON_RELEVANT: the answer does not address the question

Respond only with a valid JSON object matching this schema:
{"relevance": "RELEVANT" | "PARTLY_RELEVANT" | "NON_RELEVANT", "explanation": "<string>"}
""".strip()

judge_prompt = """
Question: {question}
Generated Answer: {answer}
""".strip()


def evaluate_relevance(question, answer, client=None, model="llama-3.1-8b-instant"):
    if client is None:
        client = Groq(api_key=GROQ_API_KEY)

    prompt = judge_prompt.format(
        question=question,
        answer=answer
    )

    # llm_structured_retry now returns a list of usage objects (one per
    # attempt, including failed ones where usage was recoverable) so
    # callers can accurately track token spend even after retries.
    result, usages = llm_structured_retry(
        client,
        judge_instructions,
        prompt,
        RelevanceVerdict,
        model=model,
    )

    return result.relevance, result.explanation


if __name__ == "__main__":
    question = "Can I still join the course?"
    answer = "Yes, you can still join. The course is self-paced."

    relevance, explanation = evaluate_relevance(question, answer)
    print(relevance)
    print(explanation)