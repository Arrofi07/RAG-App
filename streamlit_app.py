# streamlit_app.py

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from advisor_config import PROFILE_FIELDS

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Study in Germany — AI Advisor",
    page_icon="🎓",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# Session state bootstrap
# ──────────────────────────────────────────────────────────────────

for key, default in {
    "user_id":         None,
    "user_name":       None,
    "profile":         {},
    "conversation_id": None,
    "messages":        [],
    "debug_info":      [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ──────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    url = f"{API_BASE}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=30, **kwargs)
        return resp
    except Exception as e:
        st.error(f"API error: {e}")
        return None


@st.cache_data(ttl=30)
def fetch_doc_metadata():
    r = api("get", "/documents/metadata")
    return r.json().get("documents", []) if r and r.ok else []


def load_conversation(conv_id: str):
    """Load persisted messages from DB into session state."""
    r = api("get", f"/conversations/{conv_id}/messages")
    if r and r.ok:
        msgs = r.json().get("messages", [])
        st.session_state.messages   = [{"role": m["role"], "content": m["content"]} for m in msgs]
        st.session_state.debug_info = [{} for _ in msgs if m["role"] == "assistant"
                                        for m in [m]]
        # simpler: just blank debug_info per assistant turn
        assistant_count = sum(1 for m in msgs if m["role"] == "assistant")
        st.session_state.debug_info = [{} for _ in range(assistant_count)]


# ──────────────────────────────────────────────────────────────────
# ① PROFILE SETUP  (shown when no user is selected)
# ──────────────────────────────────────────────────────────────────

def show_profile_setup():
    st.title("🎓 Study in Germany — AI Advisor")
    st.subheader("Welcome! Let's set up your profile.")
    st.caption(
        "Your profile helps the advisor give you personalised, accurate advice. "
        "You can update it any time from the sidebar."
    )

    # ── Returning user selector ──────────────────────────────────────
    r = api("get", "/users")
    existing_users = r.json() if r and r.ok else []

    if existing_users:
        st.markdown("---")
        st.markdown("**Already have a profile? Select it:**")

        user_options = {u["name"]: u["id"] for u in existing_users}
        chosen_name  = st.selectbox("Your name", ["— select —"] + list(user_options))

        if chosen_name != "— select —":
            if st.button("Continue as " + chosen_name, type="primary"):
                uid  = user_options[chosen_name]
                user = api("get", f"/users/{uid}").json()
                st.session_state.user_id   = uid
                st.session_state.user_name = user["name"]
                st.session_state.profile   = user["profile"]
                st.rerun()

        st.markdown("---")
        st.markdown("**Or create a new profile:**")

    # ── New profile form ─────────────────────────────────────────────
    with st.form("profile_form"):
        name = st.text_input("Your first name *", placeholder="e.g. Budi")

        st.markdown("#### Your study situation")

        field_values: dict = {}
        cols = st.columns(2)

        for i, field in enumerate(PROFILE_FIELDS):
            col = cols[i % 2]

            with col:
                key  = field["key"]
                ftype = field["type"]

                if ftype == "select":
                    field_values[key] = col.selectbox(
                        field["label"],
                        options=field["options"],
                        index=field["options"].index(field["default"])
                        if field["default"] in field["options"] else 0,
                        help=field["help"] or None,
                    )
                elif ftype == "number":
                    field_values[key] = col.number_input(
                        field["label"],
                        min_value=0,
                        value=field["default"],
                        help=field["help"] or None,
                    )
                elif ftype == "textarea":
                    field_values[key] = st.text_area(
                        field["label"],
                        value=field["default"],
                        help=field["help"] or None,
                    )
                else:
                    field_values[key] = col.text_input(
                        field["label"],
                        value=field["default"],
                        help=field["help"] or None,
                    )

        submitted = st.form_submit_button("Save profile & start chatting 🚀", type="primary")

    if submitted:
        if not name.strip():
            st.error("Please enter your name.")
        else:
            profile         = {**field_values, "name": name.strip()}
            r               = api("post", "/users", json={"name": name.strip(), "profile": profile})

            if r and r.ok:
                user = r.json()
                st.session_state.user_id   = user["id"]
                st.session_state.user_name = user["name"]
                st.session_state.profile   = profile
                st.success(f"Welcome, {name}! Let's get started.")
                st.rerun()
            else:
                st.error("Could not create profile. Is the API running?")


# ──────────────────────────────────────────────────────────────────
# ② MAIN CHAT  (shown once a user is selected)
# ──────────────────────────────────────────────────────────────────

def show_chat():
    user_id = st.session_state.user_id

    # ── SIDEBAR ─────────────────────────────────────────────────────
    with st.sidebar:

        # Profile summary card
        profile = st.session_state.profile
        st.markdown(f"### 👤 {st.session_state.user_name}")
        st.caption(
            f"🎯 {profile.get('target_degree','?')} in {profile.get('field_of_study','?')}  \n"
            f"🌍 {profile.get('nationality','?')}  \n"
            f"🇩🇪 German: {profile.get('german_level','?')}  \n"
            f"📅 Intake: {profile.get('intake_semester','?')}"
        )

        if st.button("✏️ Edit profile", use_container_width=True):
            st.session_state.show_profile_editor = True

        # Profile editor (hidden by default)
        if st.session_state.get("show_profile_editor"):
            with st.expander("Edit your profile", expanded=True):
                with st.form("edit_profile"):
                    updated: dict = {}
                    for field in PROFILE_FIELDS:
                        key   = field["key"]
                        ftype = field["type"]
                        cur   = profile.get(key, field["default"])

                        if ftype == "select":
                            opts = field["options"]
                            idx  = opts.index(cur) if cur in opts else 0
                            updated[key] = st.selectbox(field["label"], opts, index=idx)
                        elif ftype == "number":
                            updated[key] = st.number_input(field["label"], min_value=0,
                                                            value=int(cur or field["default"]))
                        elif ftype == "textarea":
                            updated[key] = st.text_area(field["label"], value=cur or "")
                        else:
                            updated[key] = st.text_input(field["label"], value=cur or "")

                    if st.form_submit_button("Save"):
                        updated["name"] = st.session_state.user_name
                        api("put", f"/users/{user_id}/profile", json={"profile": updated})
                        st.session_state.profile              = updated
                        st.session_state.show_profile_editor = False
                        st.success("Profile updated!")
                        st.rerun()

        st.divider()

        # Conversation history
        st.markdown("### 💬 Conversations")

        if st.button("➕ New conversation", use_container_width=True):
            r = api("post", f"/users/{user_id}/conversations")
            if r and r.ok:
                st.session_state.conversation_id = r.json()["conversation_id"]
                st.session_state.messages        = []
                st.session_state.debug_info      = []
                st.rerun()

        convs_resp = api("get", f"/users/{user_id}/conversations")
        convs      = convs_resp.json() if convs_resp and convs_resp.ok else []

        for conv in convs:
            is_active = conv["id"] == st.session_state.conversation_id
            label     = ("▶ " if is_active else "") + conv["title"]
            col1, col2 = st.columns([5, 1])

            with col1:
                if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                    st.session_state.conversation_id = conv["id"]
                    load_conversation(conv["id"])
                    st.rerun()

            with col2:
                if st.button("🗑", key=f"del_{conv['id']}"):
                    api("delete", f"/conversations/{conv['id']}")
                    if st.session_state.conversation_id == conv["id"]:
                        st.session_state.conversation_id = None
                        st.session_state.messages        = []
                        st.session_state.debug_info      = []
                    st.rerun()

        st.divider()

        # Document management
        st.markdown("### 📄 Knowledge Base")

        docs_meta      = fetch_doc_metadata()
        uploaded_file  = st.file_uploader("Upload PDF", type=["pdf"])

        if uploaded_file:
            with st.expander("📝 Metadata"):
                meta_category  = st.text_input("Category")
                meta_author    = st.text_input("Author")
                meta_year      = st.number_input("Year", min_value=1900, max_value=2100, value=None)
                meta_tags      = st.text_input("Tags (comma-separated)")
            use_contextual = st.toggle("✨ Contextual enrichment", value=False,
                                       help="Slower ingest, better retrieval accuracy.")

            if st.button("⬆️ Ingest", use_container_width=True):
                uploads_dir = Path("uploads"); uploads_dir.mkdir(exist_ok=True)
                fp = uploads_dir / uploaded_file.name
                with open(fp, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Ingesting…"):
                    with open(fp, "rb") as f:
                        form_data = {
                            "category": meta_category or "",
                            "author":   meta_author   or "",
                            "tags":     meta_tags     or "",
                            "use_contextual": "true" if use_contextual else "false",
                        }
                        if meta_year:
                            form_data["year"] = str(int(meta_year))
                        resp = requests.post(f"{API_BASE}/ingest",
                                             files={"file": (uploaded_file.name, f, "application/pdf")},
                                             data=form_data)

                if resp.ok:
                    res = resp.json()
                    st.success(f"✅ {res['chunks']} chunks — `{res['source']}`")
                    st.cache_data.clear()
                else:
                    st.error(resp.text)

        with st.expander(f"📚 {len(docs_meta)} document(s)"):
            for d in docs_meta:
                st.caption(f"**{d['filename']}** | {d.get('category') or '—'}")

        st.divider()

        # Retrieval settings
        st.markdown("### ⚙️ Settings")

        top_k      = st.slider("Top K (→ LLM)",        1, 20, 5)
        fetch_k    = st.slider("Fetch K (pre-rerank)",  5, 50, 30)
        use_hybrid = st.toggle("Hybrid search", True)
        use_window = st.toggle("🪟 Window expansion", True)
        show_debug = st.toggle("Show debug info", False)

        # Metadata filters
        with st.expander("🔎 Filters"):
            filenames  = sorted({d["filename"] for d in docs_meta if d.get("filename")})
            categories = sorted({d["category"] for d in docs_meta if d.get("category")})
            f_filename = st.selectbox("File",     ["(all)"] + filenames)
            f_category = st.selectbox("Category", ["(all)"] + categories)
            active_filters: dict = {}
            if f_filename != "(all)": active_filters["filename"] = f_filename
            if f_category != "(all)": active_filters["category"] = f_category

        if st.button("🔓 Switch user", use_container_width=True):
            for k in ["user_id", "user_name", "profile", "conversation_id",
                      "messages", "debug_info"]:
                st.session_state[k] = None if k in ("user_id","user_name","conversation_id") \
                                       else ([] if isinstance(st.session_state[k], list) else {})
            st.rerun()

    # ── MAIN AREA ────────────────────────────────────────────────────

    st.title("🎓 Study in Germany — AI Advisor")

    if not st.session_state.conversation_id:
        # Auto-create a conversation on first chat
        pass

    if not st.session_state.messages:
        name = st.session_state.user_name or "there"
        st.info(
            f"Hi {name}! I'm your Study-in-Germany advisor. Ask me anything — "
            "visa process, university applications, scholarships, living costs, "
            "language requirements, or anything else about studying in Germany. "
            "Upload relevant PDFs (e.g. admission guides, scholarship brochures) "
            "to let me give you document-grounded answers."
        )

    # Render conversation
    assistant_idx = 0
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if show_debug and assistant_idx < len(st.session_state.debug_info):
                info = st.session_state.debug_info[assistant_idx]
                if info:
                    with st.expander("🔍 Debug"):
                        if info.get("rewritten_question"):
                            st.info(f"**Rewritten:** {info['rewritten_question']}")
                        flags = []
                        if info.get("window_expanded"):
                            flags.append("🪟 window expanded")
                        st.caption(
                            f"Mode: **{info.get('retrieval_mode','?')}** "
                            + (" | " + " | ".join(flags) if flags else "")
                        )
                        for j, m in enumerate(info.get("matches", []), 1):
                            score = (f"RRF `{m['rrf_score']:.4f}`" if m.get("rrf_score") is not None
                                     else f"cosine `{m.get('vector_score',0):.3f}`")
                            rerank_s = f"`{m['rerank_score']:.3f}`" if m.get("rerank_score") is not None else "—"
                            badges   = ("✨" if m.get("was_enriched") else "") + \
                                       (" 🪟" if m.get("text_sent_to_llm") and m["text_sent_to_llm"] != m["text"] else "")
                            st.markdown(f"**#{j}** *{m.get('source','')}* — {score} → {rerank_s} {badges}")
                            st.caption(m.get("text","")[:300])
            assistant_idx += 1

    # Chat input
    if prompt := st.chat_input("Ask your study-in-Germany question…"):

        # Auto-create conversation on first message
        if not st.session_state.conversation_id:
            r = api("post", f"/users/{user_id}/conversations")
            if r and r.ok:
                st.session_state.conversation_id = r.json()["conversation_id"]

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = api("post", "/query", json={
                    "question":             prompt,
                    "history":              [],          # DB is the source of truth now
                    "user_id":              user_id,
                    "conversation_id":      st.session_state.conversation_id,
                    "top_k":                top_k,
                    "fetch_k":              fetch_k,
                    "use_hybrid":           use_hybrid,
                    "use_window_expansion": use_window,
                    "filters":              active_filters,
                })

            if resp and resp.ok:
                result = resp.json()
                answer = result.get("answer", "")

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.debug_info.append({
                    "rewritten_question": result.get("rewritten_question"),
                    "retrieval_mode":     result.get("retrieval_mode"),
                    "window_expanded":    result.get("window_expanded"),
                    "sources":            result.get("sources", []),
                    "matches":            result.get("matches", []),
                })

                sources = result.get("sources", [])
                if sources:
                    st.caption("Sources: " + " · ".join(f"`{s}`" for s in sources))

                badges = []
                if result.get("rewritten_question"): badges.append("✏️ query rewritten")
                if result.get("window_expanded"):     badges.append("🪟 window expanded")
                if badges:
                    st.caption(" · ".join(badges))
            else:
                err = f"Error: {resp.text if resp else 'No response'}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.session_state.debug_info.append({})


# ──────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────

if st.session_state.user_id is None:
    show_profile_setup()
else:
    show_chat()