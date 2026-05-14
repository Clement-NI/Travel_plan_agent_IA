"""Interactive multi-turn chat using the same engine as ask.py.

Usage:
    py chat.py                         # default: Claude Sonnet 4.5
    py chat.py --ollama                # free Ollama Cloud (qwen3-coder)
    py chat.py --ollama --model gemma4:31b-cloud
    py chat.py --model claude-haiku-4-5

Why this exists: `langchain_agent_anthropic.py` / `langchain_agent_ollama.py`
(which call into `CLI.cli.run_chat_loop`) reproducibly hallucinate
phantom-failure responses ("I apologize, I cannot search for flights")
*after* a successful tool call. Verified across Gemini Flash, Claude
Sonnet 4.5, qwen3-coder:480b-cloud, gemma4:31b-cloud — all of them fail
in `run_chat_loop` and all of them succeed when invoked the same way
through `ask.py`. The exact difference is undiagnosed; somewhere in
`run_chat_loop`'s function-scope execution the model decides to bail.

This file is structurally identical to `ask.py` except the
`agent.invoke` call is wrapped in a `while True` for multi-turn — that
seems to be the structural shape that doesn't trigger the bug.
"""
from __future__ import annotations

import io
import os
import sys

# Force UTF-8 stdout on Windows so ✈ / 🚆 / 🚌 don't crash the terminal.
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from agent import load_dotenv

load_dotenv()

_args = list(sys.argv[1:])
_use_ollama = False
if "--ollama" in _args:
    _use_ollama = True
    _args.remove("--ollama")

_model = "qwen3-coder:480b-cloud" if _use_ollama else "claude-sonnet-4-5"
if "--model" in _args:
    i = _args.index("--model")
    if i + 1 < len(_args):
        _model = _args[i + 1]
        del _args[i:i + 2]
for j, a in enumerate(_args):
    if a.startswith("--model="):
        _model = a.split("=", 1)[1]
        del _args[j]
        break

if not _use_ollama and not os.environ.get("ANTHROPIC_API_KEY"):
    sys.stderr.write(
        "Set ANTHROPIC_API_KEY in .env, or pass --ollama for the free path.\n"
    )
    sys.exit(1)

import warnings
warnings.filterwarnings("ignore")

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage, HumanMessage

from CLI.cli import build_agent, _extract_text, _fmt_args
from configuration.langchain_system_prompt import SYSTEM_PROMPT
from MCP.Tools import TOOLS
from MCP import skills

if _use_ollama:
    from langchain_ollama import ChatOllama
    _model_obj = ChatOllama(
        model=_model,
        temperature=0.0,
        num_predict=8192,
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
else:
    from langchain_anthropic import ChatAnthropic
    _model_obj = ChatAnthropic(model=_model, temperature=0.0, max_tokens=4096)

_db_conn = sqlite3.connect("agent_state.db", check_same_thread=False)
agent = build_agent(_model_obj, SYSTEM_PROMPT, TOOLS, checkpointer=SqliteSaver(_db_conn))
config = {"configurable": {"thread_id": "chat"}}

print(f"Agent ready ({_model}). Type 'exit' or Ctrl-D to quit.")

# Track which tool calls we've already printed across turns so re-streamed
# messages from the checkpointer don't repeat themselves.
seen_tc_ids: set = set()
# Track which AIMessages we've already printed text from across turns.
seen_msg_ids: set = set()

while True:
    try:
        user_input = input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        break

    print("\nassistant>")

    # ---- Skills fast path ----
    # If the user's message matches a known shape (e.g.,
    # "TGV from Paris to Bordeaux on 2026-05-20"), skip the LLM and
    # call the MCP tools directly. ~5-10s instead of ~30-50s.
    #
    # IMPORTANT: we also inject the question and the skill's reply into
    # the agent's checkpointed message history so the LLM has continuity
    # on follow-ups like "of those, which is fastest?". Without this,
    # the skill turn is invisible to the agent and the next turn loses
    # context.
    import time as _time
    _t0 = _time.time()
    skill_result = skills.handle(user_input)
    if skill_result is not None:
        print(skill_result)
        print(f"\n[skill answered in {_time.time() - _t0:.1f}s — no LLM used]")
        try:
            agent.update_state(
                config,
                {"messages": [
                    HumanMessage(content=user_input),
                    AIMessage(content=skill_result),
                ]},
            )
        except Exception as exc:
            # Don't fail the turn if state update fails — the user
            # still got their answer, only follow-up context is lost.
            print(f"[warn: couldn't persist skill turn to history: {exc}]")
        continue

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
    except Exception as exc:
        print(f"[error] {exc}")
        continue

    messages = result.get("messages", [])
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        msg_id = getattr(msg, "id", None) or id(msg)
        if msg_id in seen_msg_ids:
            continue
        seen_msg_ids.add(msg_id)
        text = _extract_text(msg)
        if text:
            print(text)
        for tc in msg.tool_calls:
            tc_id = tc.get("id", id(tc))
            if tc_id in seen_tc_ids:
                continue
            seen_tc_ids.add(tc_id)
            print(f"  · {tc['name']}({_fmt_args(tc['args'])})")
