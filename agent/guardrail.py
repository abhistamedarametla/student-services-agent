from typing import Literal

from pydantic import BaseModel, Field


class AgentPlan(BaseModel):

    in_domain: bool = Field(
        description=(
            "Whether the user's request is related to "
            "education or genuine student services."
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
        description="The main category of the user's request."
    )

    action: Literal[
        "answer_directly",
        "google_search",
        "calculator",
        "reject",
    ] = Field(
        description="The action the agent should take next."
    )

    reason: str = Field(
        description=(
            "A short explanation for the classification "
            "and selected action."
        )
    )