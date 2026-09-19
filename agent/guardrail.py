from typing import Literal

from pydantic import BaseModel, Field


class DomainDecision(BaseModel):
    in_domain: bool = Field(
        description=(
            "True if the request is related to education, learning, "
            "academic support, technical learning, career development, "
            "student services, projects, internships, examinations, "
            "scholarships, resumes, interviews, or similar student needs. "
            "False if the request is unrelated."
        )
    )

    category: Literal[
        "education",
        "academic_support",
        "technical_learning",
        "career",
        "student_services",
        "other",
    ] = Field(
        description="The broad category of the user's request."
    )

    reason: str = Field(
        description=(
            "A short explanation of why the request belongs or "
            "does not belong to the student-services domain."
        )
    )


class AgentDecision(BaseModel):
    action: Literal[
        "answer_directly",
        "google_search",
        "calculator",
        "reject",
    ] = Field(
        description=(
            "The action the agent should take. "
            "answer_directly is for general knowledge or educational "
            "questions that do not require current information. "
            "google_search is for current, latest, external, or "
            "time-sensitive information. "
            "calculator is for mathematical calculations. "
            "reject is for requests outside the student-services domain."
        )
    )

    reason: str = Field(
        description="A short explanation for choosing the action."
    )