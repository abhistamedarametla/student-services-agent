from agent.guardrail import AgentDecision
from tools.calculator import (
    calculate_percentage,
    calculate_average,
)


def execute_agent_decision(
    decision: AgentDecision,
    question: str,
):
    """
    Execute the action selected by the AI agent.
    """

    if decision.action == "answer_directly":

        return {
            "type": "direct_answer",
            "question": question,
        }

    if decision.action == "google_search":

        return {
            "type": "google_search",
            "question": question,
        }

    if decision.action == "calculator":

        return {
            "type": "calculator",
            "question": question,
            "status": "calculator_ready",
        }

    if decision.action == "reject":

        return {
            "type": "rejected",
            "message": (
                "Sorry, I can only assist with "
                "educational and student-service "
                "related questions."
            ),
        }

    return {
        "type": "error",
        "message": "Unknown agent action.",
    }