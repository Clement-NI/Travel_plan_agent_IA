"""Streamlit chat UI for the travel agent.

Run with:
    py -m streamlit run app.py

(or just `streamlit run app.py` if the streamlit.exe is on your PATH)

What it does:
    - Browser chat interface at http://localhost:8501
    - Sidebar to switch provider (Anthropic / Ollama / Gemini) and model
    - Reuses the same engine as `ask.py` / `chat.py`:
        1. Regex fast-path (MCP/skills.py) tries to answer without LLM
        2. Otherwise the LangChain agent handles it, with the slim system
           prompt + on-demand `load_skill('travel')` for routing rules
    - Per-session chat history persisted in `st.session_state`
    - Per-session checkpointer so multi-turn follow-ups work
      (including the "skill answered → next turn references those results"
      flow that we shipped in `chat.py`)
"""
from __future__ import annotations

import io
import os
import sys
import time

# Force UTF-8 stdout/stderr on Windows so the ✈ 🚆 🚌 emojis don't crash
# Streamlit's subprocess logging.
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import streamlit as st

# --- Streamlit Cloud secrets → os.environ bridge ----------------------
# Locally we use `.env` (gitignored). On Streamlit Cloud there's no
# `.env`; secrets live in `st.secrets` (configured via
# Settings → Secrets in the app's dashboard, as a TOML blob). Mirror
# them into os.environ BEFORE anything else imports — the MCP server
# subprocesses inherit this environment and need the keys at spawn
# time, not after they've started.
try:
    if hasattr(st, "secrets") and len(st.secrets):
        for _k, _v in st.secrets.items():
            if isinstance(_v, (str, int, float, bool)) and not os.environ.get(_k):
                os.environ[_k] = str(_v)
except Exception:
    # `st.secrets` raises if no secrets.toml exists locally — that's
    # fine, we'll fall back to .env via load_dotenv() below.
    pass

# --- Load .env (no-op on Cloud if no .env file present) ---------------
from agent import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage

from CLI.cli import build_agent, _extract_text, _fmt_args
from configuration.langchain_system_prompt import SYSTEM_PROMPT
from MCP.Tools import TOOLS
from MCP import skills


# ---------- page setup ----------

st.set_page_config(
    page_title="Travel Agent",
    page_icon="✈",
    layout="centered",
)


# ---------- agent builder (cached so MCPs spawn once per (provider,model) pair) ----------

@st.cache_resource(show_spinner="Spinning up MCP servers…")
def _build_cached_agent(provider: str, model: str):
    """Build a LangChain agent for a given provider+model.

    Cached across reruns so we don't respawn the 4 MCP stdio subprocesses
    on every chat message — that would add ~5-10s overhead per turn.
    Streamlit re-runs the script top-to-bottom on each user input, so
    without `@st.cache_resource` the agent (and all its MCPs) would be
    rebuilt every single time.

    Each cached agent gets its OWN InMemorySaver, but we use a per-
    session thread_id so independent browser tabs don't bleed history.
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        chat = ChatAnthropic(model=model, temperature=0.0, max_tokens=4096)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        chat = ChatOllama(
            model=model, temperature=0.0, num_predict=8192,
            base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        chat = ChatGoogleGenerativeAI(model=model, temperature=0.0)
    else:
        raise ValueError(f"unknown provider {provider!r}")
    return build_agent(chat, SYSTEM_PROMPT, TOOLS, checkpointer=InMemorySaver())


# ---------- sidebar: provider + model + reset ----------

with st.sidebar:
    st.header("⚙ Settings")

    provider = st.selectbox(
        "Provider",
        options=["anthropic", "ollama", "gemini"],
        index=1,                          # default: Ollama (free)
        help="Anthropic = Claude (paid, most reliable). "
             "Ollama = free cloud or local models. "
             "Gemini = free quota, daily limit.",
    )

    default_model = {
        "anthropic": "claude-sonnet-4-5",
        "ollama":    "gpt-oss:20b-cloud",
        "gemini":    "gemini-2.5-flash",
    }[provider]
    model = st.text_input(
        "Model",
        value=default_model,
        help={
            "anthropic": "claude-sonnet-4-5 / claude-haiku-4-5 / claude-opus-4-5",
            "ollama":    "gpt-oss:20b-cloud / qwen3-coder:480b-cloud / qwen2.5:7b / ...",
            "gemini":    "gemini-2.5-flash / gemini-2.5-pro",
        }[provider],
    )

    st.markdown("---")
    if st.button("🗑  New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"ui-{int(time.time())}"
        st.rerun()

    st.markdown("---")
    st.caption(
        "Skill fast-path catches simple shapes like  \n"
        "*'TGV from Paris to Bordeaux on 2026-05-25'*  \n"
        "in ~3-5 s with no LLM call.  \n\n"
        "Ambiguous / reasoning queries fall back to the LLM."
    )


# ---------- session state ----------

if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": "...", "tag": "skill"|"llm"|None}
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    # Unique per browser session — keeps tabs/windows independent.
    st.session_state.thread_id = f"ui-{int(time.time())}"


# ---------- main: title + history ----------

st.title("✈🚆🚌 Travel Agent")
st.caption(
    f"Provider: **{provider}** / Model: `{model}` · "
    f"Thread: `{st.session_state.thread_id}`"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tag") == "skill":
            st.caption(f"⚡ answered by regex fast-path in {msg.get('elapsed','?')}s — no LLM used")
        elif msg.get("tag") == "llm":
            st.caption(f"🧠 LLM ({provider} / {model}) — {msg.get('elapsed','?')}s")


# ---------- input + answer loop ----------

user_input = st.chat_input("Ask about flights, trains, buses, or a trip plan…")

if user_input:
    # 1) Echo the user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Try the skill fast-path
    t0 = time.time()
    try:
        skill_result = skills.handle(user_input)
    except Exception as exc:
        skill_result = None
        st.warning(f"Skill router error (falling back to LLM): {exc}")

    if skill_result is not None:
        elapsed = round(time.time() - t0, 1)
        with st.chat_message("assistant"):
            st.markdown(skill_result)
            st.caption(f"⚡ skill fast-path · {elapsed}s · no LLM")
        st.session_state.messages.append({
            "role": "assistant", "content": skill_result,
            "tag": "skill", "elapsed": elapsed,
        })

        # Persist into agent state so LLM can follow up next turn
        # (same trick as in chat.py).
        try:
            agent = _build_cached_agent(provider, model)
            cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
            agent.update_state(cfg, {"messages": [
                HumanMessage(content=user_input),
                AIMessage(content=skill_result),
            ]})
        except Exception as exc:
            st.caption(f"[warn: skill turn not persisted to LLM history — {exc}]")

    else:
        # 3) LLM path
        try:
            agent = _build_cached_agent(provider, model)
        except Exception as exc:
            st.error(f"Failed to build agent: {exc}")
            st.stop()

        with st.chat_message("assistant"):
            with st.spinner(f"Asking {provider} / {model}…"):
                try:
                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )
                except Exception as exc:
                    st.error(f"[error] {exc}")
                    st.stop()

            messages = result.get("messages", [])

            # Render tool calls + final text in order
            shown_tc_ids: set = set()
            final_text = ""
            for msg in messages:
                if not isinstance(msg, AIMessage):
                    continue
                for tc in msg.tool_calls:
                    tc_id = tc.get("id", id(tc))
                    if tc_id in shown_tc_ids:
                        continue
                    shown_tc_ids.add(tc_id)
                    st.caption(f"🔧 `{tc['name']}({_fmt_args(tc['args'])})`")
                if not msg.tool_calls:
                    text = _extract_text(msg)
                    if text:
                        final_text = text

            if final_text:
                st.markdown(final_text)
            else:
                st.warning("Agent didn't produce a final text response.")

            elapsed = round(time.time() - t0, 1)
            st.caption(f"🧠 {provider} / {model} · {elapsed}s")

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_text or "_(no response)_",
            "tag": "llm",
            "elapsed": elapsed,
        })
