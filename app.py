#!/usr/bin/env python3
"""OmniPod - Chainlit UI with streaming and source cards."""

import chainlit as cl

from core.agent import run_agent


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="""# 🎙️ **OmniPod** - Podcast Knowledge Assistant

Ask me anything about the podcast transcripts! I can:

- **Factual queries**: "What did Andrej Karpathy say about AI?"
- **Multi-source synthesis**: "Compare views on meditation across guests"
- **Book/Essay generation**: "Write a comprehensive essay on consciousness"

> ⚠️ I answer **strictly** based on the podcast transcripts in my database.
""",
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    if not query:
        return

    # Thinking indicator
    thinking_msg = cl.Message(content="🤔 Thinking...")
    await thinking_msg.send()

    try:
        result = await run_agent(query)

        final_answer = result.get("final_answer", "")
        contexts = result.get("contexts", [])

        answer_msg = cl.Message(content=final_answer)

        # Only show sources if the answer actually used them
        not_discussed = "has not been discussed" in final_answer.lower()
        if contexts and not not_discussed:
            # Build source elements using Chainlit native elements
            seen = set()
            for ctx in contexts:
                key = (ctx["guest"], ctx["title"])
                if key in seen:
                    continue
                seen.add(key)

                snippet = ctx["text"][:300].replace("\n", " ")
                label = ctx["guest"]
                if ctx.get("type") == "clip":
                    label += " (clip)"

                # Add as a text element with metadata
                answer_msg.elements.append(
                    cl.Text(
                        name=label,
                        content=f'**{ctx["title"]}**\n\n> "{snippet}..."',
                        display="inline",
                        size="small",
                    )
                )

        await thinking_msg.remove()
        await answer_msg.send()

    except Exception as e:
        await thinking_msg.remove()
        await cl.Message(
            content=f"❌ **Error**: {str(e)}\n\nMake sure Qdrant is running and your API key is configured."
        ).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
