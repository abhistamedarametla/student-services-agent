import streamlit as st

from services.gemini_service import (
    analyze_request,
    continue_conversation,
    answer_with_calculation,
    answer_with_google_search,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Services Agent V2",
    page_icon="🎓",
    layout="centered",
)


# =========================================================
# PAGE TITLE
# =========================================================

st.title("🎓 Student Services Agent ")

st.caption(
    "Domain-specific AI agent for educational and student-service queries"
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Agent Capabilities")

    st.write("✅ Combined semantic domain + action analysis")
    st.write("✅ Direct educational answers")
    st.write("✅ Groq-powered calculations")
    st.write("✅ Browser Search grounding")
    st.write("✅ Conversation memory")
    st.write("✅ Out-of-domain rejection")

    st.divider()

    if st.button(
        "🔄 New Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# DISPLAY PREVIOUS CONVERSATION
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# BUILD CONVERSATION CONTEXT
# =========================================================

def build_context(question: str) -> str:
    """
    Build a compact representation of recent conversation.

    This helps the agent understand follow-up questions while
    preventing the conversation history from becoming too large.
    """

    if not st.session_state.messages:
        return question

    # Keep only the most recent 4 messages.
    recent_messages = st.session_state.messages[-4:]

    context_parts = []

    for message in recent_messages:

        role = message["role"].upper()

        # Limit each previous message to 1200 characters.
        content = message["content"][:1200]

        context_parts.append(
            f"{role}: {content}"
        )

    context_parts.append(
        f"CURRENT USER QUESTION: {question}"
    )

    context = "\n\n".join(context_parts)

    # Final safety limit for the complete context.
    return context[:6000]


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask an educational or student-service question..."
)


# =========================================================
# PROCESS USER QUESTION
# =========================================================

if user_question:

    # =====================================================
    # DISPLAY USER QUESTION
    # =====================================================

    with st.chat_message("user"):

        st.markdown(user_question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )


    # =====================================================
    # BUILD CONVERSATION CONTEXT
    # =====================================================

    contextual_question = build_context(
        user_question
    )


    try:

        # =================================================
        # STEP 1 — COMBINED AGENT ANALYSIS
        # =================================================

        with st.status(
            "🔎 Understanding and planning your request...",
            expanded=False,
        ):

            agent_plan = analyze_request(
                contextual_question
            )


        # =================================================
        # OUT-OF-DOMAIN REQUEST
        # =================================================

        if not agent_plan.in_domain:

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
                        agent_plan.category,
                    )

                    st.write(
                        "**Selected action:**",
                        agent_plan.action,
                    )

                    st.write(
                        "**Reason:**",
                        agent_plan.reason,
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
                        agent_plan.category,
                    )

                    st.write(
                        "**Selected action:**",
                        agent_plan.action,
                    )

                    st.write(
                        "**Reason:**",
                        agent_plan.reason,
                    )


                # =========================================
                # ACTION 1 — REJECT
                # =========================================

                if agent_plan.action == "reject":

                    answer = (
                        "Sorry, I can only assist with "
                        "educational and student-service "
                        "related questions."
                    )

                    st.error(answer)


                # =========================================
                # ACTION 2 — CALCULATOR
                # =========================================

                elif agent_plan.action == "calculator":

                    with st.spinner(
                        "🧮 Solving your calculation..."
                    ):

                        answer = answer_with_calculation(
                            contextual_question
                        )

                    st.markdown(answer)


                # =========================================
                # ACTION 3 — GOOGLE SEARCH
                # =========================================

                elif agent_plan.action == "google_search":

                    with st.spinner(
                        "🔎 Searching for current information..."
                    ):

                        answer = answer_with_google_search(
                            contextual_question
                        )

                    st.markdown(answer)

                    st.info(
                        "ℹ️ This answer used browser search grounding."
                    )


                # =========================================
                # ACTION 4 — DIRECT GROQ ANSWER
                # =========================================

                elif agent_plan.action == "answer_directly":

                    with st.spinner(
                        "🤖 Preparing your answer..."
                    ):

                        answer = continue_conversation(
                            question=contextual_question
                        )

                    st.markdown(answer)


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

        error_message = str(e)

        with st.chat_message("assistant"):

            # =============================================
            # GROQ RATE LIMIT ERROR
            # =============================================

            if (
                "429" in error_message
                or "rate_limit" in error_message.lower()
                or "rate limit" in error_message.lower()
                or "tokens per day" in error_message.lower()
            ):

                friendly_message = (
                    "⚠️ The AI service has temporarily reached "
                    "its usage limit. Please try again later."
                )

                st.warning(friendly_message)

                st.caption(
                    "The agent itself is working, but the external "
                    "AI service is temporarily rate-limited."
                )

                answer = friendly_message


            # =============================================
            # OTHER ERROR
            # =============================================

            else:

                answer = (
                    "⚠️ Something went wrong while processing "
                    "your request. Please try again."
                )

                st.error(answer)

                st.caption(
                    f"Technical details: {error_message}"
                )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )