"""
Rubric-based judge for Levi's RAG answers.

Design choice: the judge model should NOT be the same model that generated
the answer (Gemini). Using a different model as judge avoids self-preference
bias, a documented issue in LLM-as-judge setups (see Zheng et al., MT-Bench
paper). This file uses Groq's free-tier API as the judge, since it's zero
cost and you already have a working Groq setup from the ASD project.
Check Groq's current model list before running (model names get deprecated
and replaced) — swap JUDGE_MODEL below if llama-3.3-70b-versatile is no
longer available.
"""

import json
from groq import Groq

client = Groq()  # reads GROQ_API_KEY from env
JUDGE_MODEL = "llama-3.3-70b-versatile"

RUBRIC_PROMPT = """You are a strict legal-document QA evaluator. You will be given:
1. A document excerpt (the only source of truth the assistant was allowed to use)
2. A question asked about that excerpt
3. An answer produced by an AI assistant

Score the answer on two 1-5 scales and one boolean flag. Be strict: do not
give credit for answers that sound confident but add information not in
the excerpt.

- faithfulness (1-5): 5 = every claim in the answer is directly supported
  by the excerpt. 1 = the answer contradicts or invents facts not in the
  excerpt.
- completeness (1-5): 5 = the answer covers every relevant point in the
  excerpt that the question asks about. 1 = the answer misses most of the
  relevant content.
- fabricated (true/false): true if the answer states any fact, number,
  clause, or obligation that is NOT present in the excerpt.

Respond with ONLY valid JSON, no markdown, no explanation outside the JSON:
{{"faithfulness": <int 1-5>, "completeness": <int 1-5>, "fabricated": <true|false>, "reasoning": "<one sentence>"}}

DOCUMENT EXCERPT:
{document_context}

QUESTION:
{query}

ASSISTANT ANSWER:
{answer}
"""


def judge_answer(document_context: str, query: str, answer: str) -> dict:
    prompt = RUBRIC_PROMPT.format(
        document_context=document_context, query=query, answer=answer
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    # some Groq models wrap JSON in markdown fences even when told not to
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)
