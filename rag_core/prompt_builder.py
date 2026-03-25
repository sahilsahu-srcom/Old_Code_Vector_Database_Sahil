"""
File Summary:
Assembles the final list of Messages sent to the LLM for the HAUP RAG engine.
Injects retrieved context only into the current user turn to save tokens.
Supports schema summary in system prompt and graceful fallback when no records found.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

PromptBuilder()  [Class → Object]
||
├── __init__()  [Function] -------------------------------> Store schema_summary string
│
├── build()  [Function] ----------------------------------> Assemble complete message list for LLM
│       │
│       ├── Build system message -------------------------> _SYSTEM_TEMPLATE + optional schema section
│       │       │
│       │       └── [Conditional Branch] schema_summary --> Inject _SCHEMA_SECTION if provided
│       │
│       ├── Append conversation history ------------------> Prior turns without context (saves tokens)
│       │
│       ├── [Conditional Branch] has_results and context -> Inject _CONTEXT_INJECTION into user message
│       │       │
│       │       └── True  --------------------------------> Format retrieved records + question
│       │
│       └── [Conditional Branch] no results / no context -> Use _NO_CONTEXT_INJECTION fallback
│               │
│               └── False --------------------------------> Inform LLM no records found
│
└── update_schema()  [Function] --------------------------> Update schema_summary at runtime

Message layout sent to LLM:
  [system]    Role definition + schema + instructions
  [user]      Turn 1
  [assistant] Turn 1 reply
  ...
  [user]      Current turn WITH injected context block

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

from typing import List, Optional

from rag_core.llm_client import Message


_SYSTEM_TEMPLATE = """\
You are a data analyst assistant with access to a structured database. \
Your job is to answer user questions accurately by analysing the retrieved \
database records provided in each message.

Guidelines:
- Base your answer ONLY on the retrieved records shown below.
- If the records do not contain enough information to answer, say so clearly.
- Be concise and factual. Avoid speculation.
- When referencing specific rows, use their citation number [N].
- If asked to count, aggregate, or filter, do so based on the visible records.
- Never invent or hallucinate data that is not in the provided records.
- If the user asks a follow-up question, use the conversation history for context.

{schema_section}
"""

_SCHEMA_SECTION = """\
Database schema context:
{schema_summary}
"""

_CONTEXT_INJECTION = """\
--- Retrieved Database Records ---
{context}
--- End of Records ---

User question: {question}"""

_NO_CONTEXT_INJECTION = """\
No matching records were found in the database for this query.
Please answer based on conversation history if relevant, otherwise \
indicate that the information is not available.

User question: {question}"""


"""================= Startup class PromptBuilder ================="""
class PromptBuilder:

    """================= Startup method __init__ ================="""
    def __init__(self, schema_summary: str = ""):
        self._schema_summary = schema_summary
    """================= End method __init__ ================="""

    """================= Startup method build ================="""
    def build(
        self,
        question:    str,
        context:     str,
        history:     List[Message],
        *,
        has_results: bool = True,
    ) -> List[Message]:
        """
        Returns the complete message list to send to the LLM.

        Args:
            question:    Current user question (raw text, no context injected yet).
            context:     Formatted context string from ContextBuilder.
            history:     Previous turns from ConversationManager.to_messages().
            has_results: False if retrieval returned nothing.
        """
        # 1. System message
        schema_section = ""
        if self._schema_summary:
            schema_section = _SCHEMA_SECTION.format(schema_summary=self._schema_summary)
        system_content = _SYSTEM_TEMPLATE.format(schema_section=schema_section).strip()

        messages: List[Message] = [Message(role="system", content=system_content)]

        # 2. Conversation history (older turns WITHOUT context injected)
        messages.extend(history)

        # 3. Current turn WITH context injected into user message
        if has_results and context:
            user_content = _CONTEXT_INJECTION.format(
                context  = context,
                question = question,
            )
        else:
            user_content = _NO_CONTEXT_INJECTION.format(question=question)

        messages.append(Message(role="user", content=user_content))

        return messages
    """================= End method build ================="""

    """================= Startup method update_schema ================="""
    def update_schema(self, schema_summary: str) -> None:
        self._schema_summary = schema_summary
    """================= End method update_schema ================="""

"""================= End class PromptBuilder ================="""