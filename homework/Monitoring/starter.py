"""Starter code for the monitoring homework.

Sets up the text-search RAG from homework 1 and a shared OpenAI client.
"""

from groq import Groq

from gitsource import GithubRepositoryDataReader
from minsearch import Index

from rag_helper import RAGBase
from dotenv import load_dotenv
load_dotenv()
import os

COMMIT = "8c1834d"

# --- Load the course lessons (same as HW1, HW2, HW4) ---
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
rag = RAGBase(index=index, llm_client=client)

if __name__ == "__main__":
    query = "How does the agentic loop keep calling the model until it stops?"

    search_results = rag.search(query)
    print("NUM RESULTS:", len(search_results))
    print("FIRST RESULT KEYS:", search_results[0].keys() if search_results else None)

    prompt = rag.build_prompt(query, search_results)
    print("---PROMPT---")
    print(prompt[:1000])   # just the first part, to eyeball it
    print("---END PROMPT---")

    response = rag.llm(prompt)
    print("RAW RESPONSE:", response)

    answer = response.choices[0].message.content
    print("ANSWER:", answer)
