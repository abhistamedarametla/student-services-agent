import streamlit as st

from google import genai
from google.genai import types

from agent.guardrail import DomainDecision, AgentDecision


# ============================================================
# DOMAIN CLASSIFICATION INSTRUCTIONS
# ============================================================

DOMAIN_INSTRUCTIONS = """
You are the domain classifier for a Student Services AI Agent.

The system assists students with educational and student-service needs.

Determine whether the user's request is primarily related to:
education, learning, academic support, technical learning,
programming, computer science, AI/ML, academic projects,
career development, placement preparation, internships,
examinations, scholarships, resumes, interviews,
or other genuine student-service needs.

Understand the meaning and purpose of the complete request.
Do not classify using individual keywords alone.

Examples:

1. "Who won yesterday's cricket match?"
   -> Outside the domain.

2. "I am doing a college project on sports analytics.
   Explain how machine learning can predict cricket player
   performance."
   -> In the domain because the purpose is educational/
      technical learning.

If the request is unrelated to education or student services,
mark it as outside the domain.

Return only the structured result requested by the response schema.
"""


# ============================================================
# ACTION PLANNING INSTRUCTIONS
# ============================================================

PLANNER_INSTRUCTIONS = """
You are the decision-making component of a Student Services AI Agent.

The user has already passed the student-services domain check.

Decide what the agent should do next.

Choose exactly one action:

1. answer_directly
   Use for general educational, academic, programming,
   technical, career, or student-service questions that
   can be answered from general knowledge.

2. google_search
   Use when the user needs current, latest, recent,
   live, changing, or externally verifiable information.

3. calculator
   Use when the main task is a mathematical or numerical
   calculation.

4. reject
   Use only if the request is actually outside the
   student-services domain.

Understand the complete meaning of the request.
Do not choose an action based only on keywords.

Return only the structured result requested by the response schema.
"""


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():
    """
    Create and cache the Gemini API client.
    """

    api_key = st.secrets["GEMINI_API_KEY"]

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            api_version="v1",
            timeout=60000,
        ),
    )


# ============================================================
# DOMAIN CLASSIFICATION
# ============================================================

def classify_domain(question: str) -> DomainDecision:
    """
    Determine whether the user's question belongs
    to the student-services domain.
    """

    client = get_gemini_client()

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        system_instruction=DOMAIN_INSTRUCTIONS,
        input=question,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": DomainDecision.model_json_schema(),
        },
        generation_config={
            "thinking_level": "low",
        },
    )

    return DomainDecision.model_validate_json(
        interaction.output_text
    )


# ============================================================
# ACTION PLANNING
# ============================================================

def plan_action(question: str) -> AgentDecision:
    """
    Decide what action the agent should perform
    for an in-domain question.
    """

    client = get_gemini_client()

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        system_instruction=PLANNER_INSTRUCTIONS,
        input=question,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": AgentDecision.model_json_schema(),
        },
        generation_config={
            "thinking_level": "low",
        },
    )

    return AgentDecision.model_validate_json(
        interaction.output_text
    )


# ============================================================
# GOOGLE SEARCH
# ============================================================

def answer_with_google_search(question: str) -> str:
    """
    Answer a question using Gemini with Google Search grounding.
    """

    client = get_gemini_client()

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=question,
        tools=[
            {
                "type": "google_search"
            }
        ],
        generation_config={
            "thinking_level": "low",
        },
    )

    return interaction.output_text


# ============================================================
# CONTINUOUS CONVERSATION
# ============================================================

def continue_conversation(
    question: str,
    previous_interaction_id: str | None = None,
):
    """
    Continue a multi-turn conversation with Gemini.

    If previous_interaction_id is provided,
    Gemini can use the previous interaction as context.
    """

    client = get_gemini_client()

    if previous_interaction_id is not None:

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=question,
            previous_interaction_id=previous_interaction_id,
            generation_config={
                "thinking_level": "low",
            },
        )

    else:

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=question,
            generation_config={
                "thinking_level": "low",
            },
        )

    return interaction