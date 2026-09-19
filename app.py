import streamlit as st

from services.gemini_service import (
    classify_domain,
    plan_action,
    continue_conversation,
    answer_with_google_search,
)

from services.agent_service import execute_agent_decision


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Services Agent",
    page_icon="🎓",
    layout="centered",
)


st.title("🎓 Student Services Agent")

st.caption(
    "Domain-specific AI agent for educational and student-service queries"
)


# =========================================================
# SESSION STATE
# =========================================================

if "previous_interaction_id" not in st.session_state:
    st.session_state.previous_interaction_id = None


if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Agent Capabilities")

    st.write("✅ Semantic domain classification")
    st.write("✅ Agent action planning")
    st.write("✅ Direct educational answers")
    st.write("✅ Gemini-based calculations")
    st.write("✅ Google Search grounding")
    st.write("✅ Conversation memory")
    st.write("✅ Out-of-domain rejection")

    st.divider()

    if st.button(
        "🔄 New Conversation",
        use_container_width=True,
    ):

        st.session_state.previous_interaction_id = None
        st.session_state.messages = []

        st.rerun()


# =========================================================
# DISPLAY PREVIOUS CONVERSATION
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# =========================================================
# BUILD CONVERSATION CONTEXT
# =========================================================

def build_context(question: str) -> str:
    """
    Build a small representation of recent conversation.

    This helps the domain classifier and planner understand
    follow-up questions such as:

    "Give me an example."
    "What are its advantages?"
    "Calculate that."
    """

    if not st.session_state.messages:
        return question

    recent_messages = st.session_state.messages[-6:]

    context_parts = []

    for message in recent_messages:

        role = message["role"].upper()

        content = message["content"]

        context_parts.append(
            f"{role}: {content}"
        )

    context_parts.append(
        f"CURRENT USER QUESTION: {question}"
    )

    return "\n\n".join(context_parts)


# =========================================================
# GEMINI CALCULATION
# =========================================================

def answer_with_calculation(question: str) -> str:
    """
    Ask Gemini to understand and solve a calculation-related
    student question.

    The calculation is not restricted to predefined functions
    such as percentage() or average().

    Gemini determines what mathematical calculation is required
    from the natural-language question.
    """

    calculation_instruction = """
You are the calculation component of a Student Services AI Agent.

Solve the user's calculation question carefully.

The question may involve any reasonable mathematical calculation
related to education, academics, examinations, marks, percentages,
averages, ratios, weighted scores, attendance, grades, comparisons,
or other numerical student-related problems.

Do NOT assume that the question must match a predefined calculation
type.

Understand the complete meaning of the question first.

Show the important calculation steps clearly.

Check the arithmetic before giving the final answer.

If necessary information is missing, clearly state what information
is required instead of inventing values.

If the question is ambiguous, explain the assumption you are making.

Return a clear, student-friendly answer.
"""

    from google import genai
    from google.genai import types

    client = st.session_state.get("_gemini_client")

    if client is None:

        api_key = st.secrets["GEMINI_API_KEY"]

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                api_version="v1",
                timeout=60000,
            ),
        )

        st.session_state["_gemini_client"] = client

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        system_instruction=calculation_instruction,
        input=question,
        generation_config={
            "thinking_level": "low",
        },
    )

    return interaction.output_text


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask an educational or student-service question..."
)


if user_question:

    # =====================================================
    # DISPLAY USER QUESTION
    # =====================================================

    with st.chat_message("user"):

        st.write(user_question)


    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )


    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    contextual_question = build_context(
        user_question
    )


    try:

        # =================================================
        # STEP 1 — SEMANTIC DOMAIN CLASSIFICATION
        # =================================================

        with st.status(
            "🔎 Understanding your request...",
            expanded=False,
        ):

            domain_decision = classify_domain(
                contextual_question
            )


        # =================================================
        # OUT-OF-DOMAIN REQUEST
        # =================================================

        if not domain_decision.in_domain:

            answer = (
                "Sorry, I can only assist with "
                "educational and student-service "
                "related questions."
            )

            with st.chat_message("assistant"):

                st.error(answer)

                with st.expander(
                    "🔍 Agent decision details"
                ):

                    st.write(
                        "**Domain:** Outside domain"
                    )

                    st.write(
                        "**Category:**",
                        domain_decision.category,
                    )

                    st.write(
                        "**Reason:**",
                        domain_decision.reason,
                    )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


        # =================================================
        # IN-DOMAIN REQUEST
        # =================================================

        else:

            # =============================================
            # STEP 2 — AGENT PLANNING
            # =============================================

            with st.status(
                "🧠 Planning the best action...",
                expanded=False,
            ):

                agent_decision = plan_action(
                    contextual_question
                )


            # =============================================
            # ASSISTANT RESPONSE
            # =============================================

            with st.chat_message("assistant"):

                # =========================================
                # AGENT DECISION DETAILS
                # =========================================

                with st.expander(
                    "🔍 Agent decision details"
                ):

                    st.write(
                        "**Domain:** In domain"
                    )

                    st.write(
                        "**Category:**",
                        domain_decision.category,
                    )

                    st.write(
                        "**Selected action:**",
                        agent_decision.action,
                    )

                    st.write(
                        "**Reason:**",
                        agent_decision.reason,
                    )


                    # =====================================
                    # EXECUTION LAYER
                    # =====================================

                    execution_result = (
                        execute_agent_decision(
                            agent_decision,
                            user_question,
                        )
                    )

                    st.write(
                        "**Execution type:**",
                        execution_result["type"],
                    )


                # =========================================
                # ACTION 1 — REJECT
                # =========================================

                if agent_decision.action == "reject":

                    answer = (
                        "Sorry, I can only assist with "
                        "educational and student-service "
                        "related questions."
                    )

                    st.error(answer)


                # =========================================
                # ACTION 2 — CALCULATOR
                # =========================================

                elif (
                    agent_decision.action
                    == "calculator"
                ):

                    with st.spinner(
                        "🧮 Solving your calculation..."
                    ):

                        answer = (
                            answer_with_calculation(
                                contextual_question
                            )
                        )

                    st.write(answer)


                # =========================================
                # ACTION 3 — GOOGLE SEARCH
                # =========================================

                elif (
                    agent_decision.action
                    == "google_search"
                ):

                    with st.spinner(
                        "🔎 Searching for current information..."
                    ):

                        answer = (
                            answer_with_google_search(
                                contextual_question
                            )
                        )

                    st.write(answer)

                    st.info(
                        "ℹ️ This answer used "
                        "Google Search grounding."
                    )


                # =========================================
                # ACTION 4 — DIRECT GEMINI ANSWER
                # =========================================

                elif (
                    agent_decision.action
                    == "answer_directly"
                ):

                    with st.spinner(
                        "🤖 Preparing your answer..."
                    ):

                        interaction = (
                            continue_conversation(
                                question=user_question,
                                previous_interaction_id=(
                                    st.session_state
                                    .previous_interaction_id
                                ),
                            )
                        )

                    answer = interaction.output_text

                    st.write(answer)


                    # =====================================
                    # SAVE CONVERSATION ID
                    # =====================================

                    st.session_state.previous_interaction_id = (
                        interaction.id
                    )


                # =========================================
                # UNKNOWN ACTION
                # =========================================

                else:

                    answer = (
                        "I couldn't determine the "
                        "appropriate action for this request."
                    )

                    st.warning(answer)


            # =================================================
            # SAVE ASSISTANT RESPONSE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


    except Exception as e:

        with st.chat_message("assistant"):

            st.error(
                f"Something went wrong: {e}"
            )