import time

from tqdm.auto import tqdm
from rag_helper import RAGBase
from pydantic import BaseModel

class Questions(BaseModel):
    questions: list[str]


class LLMStructuredError(Exception):
    """Raised when a structured LLM call fails, carrying usage if available."""

    def __init__(self, original_exception, usage=None):
        super().__init__(str(original_exception))
        self.original_exception = original_exception
        self.usage = usage


def calc_price(usage):
    input_price_per_million = 0.75
    output_price_per_million = 4.50

    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured(client, instructions, user_prompt, response_model=Questions, model="llama-3.1-8b-instant"):
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    try:
        parsed = response_model.model_validate_json(
            response.choices[0].message.content
        )
    except Exception as e:
        # Preserve usage even when parsing the response fails, so callers
        # can still account for the tokens that were spent on this attempt.
        raise LLMStructuredError(e, usage=response.usage) from e

    return parsed, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    response_model=Questions,
    model="llama-3.1-8b-instant",
    max_retries=3,
):
    usages = []
    last_exception = None

    for attempt in range(max_retries):
        try:
            parsed, usage = llm_structured(
                client,
                instructions,
                user_prompt,
                response_model=response_model,
                model=model,
            )
            usages.append(usage)
            return parsed, usages
        except LLMStructuredError as e:
            if e.usage is not None:
                usages.append(e.usage)
            last_exception = e.original_exception
        except Exception as e:
            last_exception = e

        if attempt == max_retries - 1:
            raise last_exception

        time.sleep(2 ** attempt)

    # Unreachable: the loop above always either returns or raises.
    raise last_exception


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=input_messages
            )
        return response.choices[0].message.content.strip()


    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results