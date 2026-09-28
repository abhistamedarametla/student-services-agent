import json

import streamlit as st
from groq import Groq

from agent.guardrail import AgentPlan


MODEL_NAME = "openai/gpt-oss-20b"


@st.cache_resource
def get_groq_client():
    """
    Create one reusable Groq client.
    The API key is stored in Streamlit secrets.
    """
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def analyze_request(question: str) -> AgentPlan:
    """
    Decide:
    1. Is the request inside the student-services domain?
    2. What category does it belong to?
    3. Which action should the agent perform?

    One Groq call performs all three decisions.
    """

    client = get_groq_client()

    question = question[-6000:]

    system_prompt = """
You are the planner for a student-services AI agent.

Allowed domain:
- education
- academic support
- programming and technical learning
- AI/ML
- projects
- careers, internships and placements
- student services

Reject unrelated topics such as:
- sports results
- weather
- entertainment
- celebrities
- unrelated shopping
- unrelated travel
- unrelated politics

Important:
Being a student does not automatically make an unrelated question
part of the domain. Judge the actual purpose of the request.

Choose one action:

answer_directly
- Stable conceptual educational questions.
- Example: Explain Java inheritance.

google_search
- Current, latest, recent or externally changing information.
- Example: Find current AI internships in India.
- Also use this for specific external information such as:
  college locations, official contact details, current courses,
  current admission information, current deadlines, etc.

calculator
- Numerical or academic calculations.
- Example: Calculate 437 out of 550 as a percentage.
- Calculator questions use category academic_support.

reject
- Clearly outside the allowed domain.

Categories:
education
academic_support
technical_learning
career
student_services
other

Return only the AgentPlan structured output.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "agent_plan",
                "schema": AgentPlan.model_json_schema(),
            },
        },
    )

    content = response.choices[0].message.content

    data = json.loads(content)

    return AgentPlan.model_validate(data)


def continue_conversation(
    question: str,
    previous_interaction_id=None,
):
    """
    Generate a direct educational answer.
    """

    client = get_groq_client()

    question = question[-8000:]

    system_prompt = """
You are a student-services AI assistant.

Help with:
- education
- academics
- programming
- computer science
- AI/ML
- projects
- careers
- internships
- placement preparation

Give accurate, clear and beginner-friendly answers.

For conceptual questions, answer directly.

Do not invent facts.

Use examples when they improve understanding.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response.choices[0].message.content


def answer_with_calculation(question: str):
    """
    Solve an academic or numerical calculation.

    The response uses simple Markdown/plain text
    instead of LaTeX so it displays cleanly in Streamlit.
    """

    client = get_groq_client()

    question = question[-5000:]

    system_prompt = """
You are the calculation component of a student-services AI agent.

Solve the numerical problem accurately.

Show:

1. Formula or method
2. Important calculation steps
3. Final answer

IMPORTANT FORMATTING RULES:

- Do NOT use LaTeX.
- Do NOT use \\[ ... \\].
- Do NOT use \\( ... \\).
- Do NOT use LaTeX commands such as \\frac, \\text, \\boxed,
  \\times, or \\%.
- Use simple readable text and normal Markdown.
- You may use normal mathematical symbols such as:
  ×, ÷, =, +, -, and %.
- Avoid unnecessarily long decimal expansions.
- Round the final answer appropriately.

Example:

Formula:
Percentage = (Part ÷ Whole) × 100

Calculation:
(437 ÷ 550) × 100 = 79.45%

Final Answer:
**79.45%**

Check arithmetic before responding.

If information is missing, say what is required.

Do not invent values.

Keep the explanation concise.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response.choices[0].message.content


def answer_with_google_search(question: str):
    """
    Answer current/external questions using Groq Browser Search.

    Browser Search provides real-time web grounding.
    Raw citation markers are disabled so the Streamlit UI
    does not display markers such as [1†L6-L11].
    """

    client = get_groq_client()

    question = question[-4000:]

    system_prompt = """
You are the web-search component of a Student Services AI Agent.

Use Browser Search to find current and reliable information.

Accuracy rules:

- Prefer official or authoritative sources.
- For colleges, prefer the official college website when available.
- For current/latest/recent questions, verify dates.
- Do not present expired information as current.
- Do not invent companies, internships, deadlines, salaries,
  eligibility, dates, addresses or URLs.
- If a detail cannot be verified, clearly say so.
- If sources conflict, mention the conflict.
- For internships/jobs, prefer official career/application pages.
- Answer only the user's question.
- Keep the final answer concise and useful.

Important formatting rule:

- Do not write raw citation markers such as:
  [1†L6-L11]
  [2†L20-L25]
- Do not include citation-marker notation in the final answer.
- Present the answer naturally in clean Markdown.

Use Browser Search before producing the answer.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        tools=[
            {
                "type": "browser_search"
            }
        ],
        tool_choice="required",

        # Prevent Groq's raw web citation markers
        # from appearing in the Streamlit answer.
        citation_options="disabled",

        reasoning_effort="low",
        max_completion_tokens=1024,
    )

    return response.choices[0].message.content