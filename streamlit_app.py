# streamlit_app.py  (v8.0.0)
#
# Streamlit frontend for the Study-in-Germany AI Advisor.
#
# V8 CHANGES over v7:
# ─────────────────────────────────────────────────────────────────────────────
# 1. INTENT BADGE  — each assistant reply now shows a small badge indicating
#    which plan the agentic planner used (RAG / Web Search / University Match).
#    Helps students understand why the answer looks the way it does.
#
# 2. WEB SOURCES PANEL — if the planner triggered a web search, the assistant
#    message expands to show the live URLs that were searched.
#
# 3. UNIVERSITY FINDER PAGE — a new sidebar tab calls POST /recommend directly
#    from the student's saved profile.  Results are displayed as cards with
#    key facts (city, fee, language, deadlines) and a match score bar.
#
# 4. FORCE INTENT OPTION — power users can pin the intent from the chat input
#    area (e.g. "always use web search" toggle for time-sensitive sessions).

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

# ─────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────

for key, default in {
    "user_id":         None,
    "user_name":       None,
    "profile":         {},
    "conversation_id": None,
    "messages":        [],
    "debug_info":      [],
    "active_page":     "chat",  # "chat" | "universities" | "profile"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    """Make an API call to the FastAPI backend. Returns the Response or None."""
    url = f"{API_BASE}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=60, **kwargs)
        return resp
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None


@st.cache_data(ttl=30)
def fetch_doc_metadata():
    """Cached fetch of ingested document metadata for filter dropdowns."""
    r = api("get", "/documents/metadata")
    return r.json().get("documents", []) if r and r.ok else []


def load_conversation(conv_id: str):
    """Load persisted messages from DB into session state."""
    r = api("get", f"/conversations/{conv_id}/messages")
    if r and r.ok:
        msgs = r.json().get("messages", [])
        st.session_state.messages   = [{"role": m["role"], "content": m["content"]} for m in msgs]
        # Reset debug info (one entry per assistant turn)
        assistant_count = sum(1 for m in msgs if m["role"] == "assistant")
        st.session_state.debug_info = [{} for _ in range(assistant_count)]


# ─────────────────────────────────────────────────────────────────────────────
# Intent badge helper
# ─────────────────────────────────────────────────────────────────────────────

# Map planner intent strings → (emoji, label, colour)
_INTENT_LABELS = {
    "rag_only":               ("📄", "Knowledge Base",      "#3b82f6"),
    "web_search":             ("🔍", "Live Web Search",     "#10b981"),
    "university_recommender": ("🏫", "University Finder",   "#8b5cf6"),
    "recommend_and_search":   ("🔍🏫", "Finder + Web",       "#f59e0b"),
}


def _intent_badge(intent: str) -> str:
    """Return an HTML span badge for the given intent."""
    emoji, label, colour = _INTENT_LABELS.get(
        intent, ("⚙️", intent, "#6b7280")
    )
    return (
        f'<span style="background:{colour}22; color:{colour}; border:1px solid {colour}; '
        f'border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600;">'
        f'{emoji} {label}</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# ① PROFILE SETUP (shown before user is created)
# ─────────────────────────────────────────────────────────────────────────────

def show_profile_setup():
    st.title("🎓 Study in Germany — AI Advisor")
    st.subheader("Welcome! Let's set up your profile.")
    st.caption(
        "Your profile lets the advisor give you personalised, accurate advice. "
        "You can update it any time from the sidebar."
    )

    # Check for existing users so returning visitors don't need to re-register
    r = api("get", "/users")
    existing = r.json() if r and r.ok else []

    if existing:
        with st.expander("👤 Continue as existing user"):
            selected = st.selectbox(
                "Select your name",
                options=existing,
                format_func=lambda u: u["name"],
                key="existing_user_select",
            )
            if st.button("Continue", key="continue_existing"):
                _set_user(selected["id"], selected["name"])
                st.rerun()

    st.divider()
    st.subheader("Create a new profile")

    with st.form("new_profile_form"):
        name = st.text_input("Your name *", placeholder="e.g. Budi Santoso")

        profile_data: dict = {}
        # Render profile fields dynamically from advisor_config.py
        for field in PROFILE_FIELDS:
            key   = field["key"]
            label = field["label"]
            help_ = field.get("help", "")

            if field["type"] == "text":
                profile_data[key] = st.text_input(label, help=help_,
                                                   value=field.get("default", ""))
            elif field["type"] == "select":
                profile_data[key] = st.selectbox(label, options=field["options"],
                                                  help=help_,
                                                  index=field["options"].index(field["default"])
                                                  if field.get("default") in field["options"] else 0)
            elif field["type"] == "number":
                profile_data[key] = st.number_input(label, help=help_,
                                                     value=field.get("default", 0),
                                                     min_value=0)
            elif field["type"] == "textarea":
                profile_data[key] = st.text_area(label, help=help_,
                                                  value=field.get("default", ""))

        submitted = st.form_submit_button("✅ Save and start chatting")

    if submitted:
        if not name.strip():
            st.error("Please enter your name to continue.")
            return

        r = api("post", "/users", json={"name": name.strip(), "profile": profile_data})
        if r and r.ok:
            user = r.json()
            _set_user(user["id"], user["name"])

            # Start the first conversation immediately
            rc = api("post", f"/users/{user['id']}/conversations",
                     params={"title": "First conversation"})
            if rc and rc.ok:
                st.session_state.conversation_id = rc.json()["conversation_id"]

            st.success("Profile saved! Starting your session…")
            st.rerun()
        else:
            st.error("Could not create profile. Is the backend running?")


def _set_user(user_id: str, user_name: str):
    """Store the logged-in user's identity in session state."""
    st.session_state.user_id   = user_id
    st.session_state.user_name = user_name
    # Fetch the full profile so sidebar fields are pre-filled
    r = api("get", f"/users/{user_id}")
    if r and r.ok:
        st.session_state.profile = r.json().get("profile", {})


# ─────────────────────────────────────────────────────────────────────────────
# ② SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user_name}")

        # ── Page navigation ────────────────────────────────────────────────────
        st.divider()
        if st.button("💬 Chat with Advisor", use_container_width=True):
            st.session_state.active_page = "chat"
            st.rerun()

        if st.button("🏫 Find Universities", use_container_width=True):
            st.session_state.active_page = "universities"
            st.rerun()

        if st.button("👤 Edit My Profile", use_container_width=True):
            st.session_state.active_page = "profile"
            st.rerun()

        # ── Conversation management ────────────────────────────────────────────
        st.divider()
        st.markdown("**Conversations**")

        r = api("get", f"/users/{st.session_state.user_id}/conversations")
        convs = r.json() if r and r.ok else []

        if st.button("➕ New conversation", use_container_width=True):
            rc = api("post", f"/users/{st.session_state.user_id}/conversations")
            if rc and rc.ok:
                cid = rc.json()["conversation_id"]
                st.session_state.conversation_id = cid
                st.session_state.messages        = []
                st.session_state.debug_info      = []
                st.session_state.active_page     = "chat"
                st.rerun()

        for conv in convs:
            col1, col2 = st.columns([4, 1])
            is_active  = conv["id"] == st.session_state.conversation_id
            label      = ("▶ " if is_active else "") + conv["title"][:35]

            if col1.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                st.session_state.conversation_id = conv["id"]
                st.session_state.active_page     = "chat"
                load_conversation(conv["id"])
                st.rerun()

            if col2.button("🗑", key=f"del_{conv['id']}"):
                api("delete", f"/conversations/{conv['id']}")
                if st.session_state.conversation_id == conv["id"]:
                    st.session_state.conversation_id = None
                    st.session_state.messages        = []
                    st.session_state.debug_info      = []
                st.rerun()

        # ── Document upload (admin section) ────────────────────────────────────
        st.divider()
        st.markdown("**📄 Upload Knowledge Documents**")

        with st.expander("Upload PDF"):
            _render_upload_form()

        # ── App version ────────────────────────────────────────────────────────
        st.divider()
        st.caption("v8.0.0 — Agentic Advisor")


def _render_upload_form():
    """PDF upload form in the sidebar."""
    uploaded = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")
    if not uploaded:
        return

    category = st.text_input("Category (optional)", key="cat_inp")
    author   = st.text_input("Author (optional)",   key="auth_inp")
    year     = st.number_input("Year (optional)", min_value=2000, max_value=2030,
                                value=2024, key="year_inp")
    tags     = st.text_input("Tags (comma-separated)", key="tags_inp")
    use_ctx  = st.checkbox("Contextual enrichment (slower, smarter)", value=False)

    if st.button("⬆️ Ingest PDF"):
        with st.spinner("Ingesting…"):
            r = api(
                "post", "/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                data={
                    "category":       category or "",
                    "author":         author or "",
                    "year":           str(year),
                    "tags":           tags or "",
                    "use_contextual": "true" if use_ctx else "false",
                },
            )
        if r and r.ok:
            res = r.json()
            st.success(f"✅ {res['chunks']} chunks ingested from {res['source']}")
            st.cache_data.clear()
        else:
            st.error("Ingestion failed. Check backend logs.")


# ─────────────────────────────────────────────────────────────────────────────
# ③ CHAT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def show_chat_page():
    """Main chat interface with intent badge and web source display."""
    st.title("💬 AI Advisor Chat")

    # Ensure a conversation exists
    if not st.session_state.conversation_id:
        rc = api("post", f"/users/{st.session_state.user_id}/conversations")
        if rc and rc.ok:
            st.session_state.conversation_id = rc.json()["conversation_id"]

    # ── Message history ────────────────────────────────────────────────────────
    debug_idx = 0
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show debug info for assistant messages
            if msg["role"] == "assistant" and debug_idx < len(st.session_state.debug_info):
                info = st.session_state.debug_info[debug_idx]
                debug_idx += 1

                # Intent badge
                intent = info.get("intent")
                if intent:
                    st.markdown(_intent_badge(intent), unsafe_allow_html=True)

                # Web sources (if any)
                web_sources = info.get("web_sources", [])
                if web_sources:
                    with st.expander(f"🔍 {len(web_sources)} live web source(s) checked"):
                        for src in web_sources:
                            title = src.get("title") or src.get("url", "")
                            url   = src.get("url", "")
                            if url:
                                st.markdown(f"- [{title}]({url})")
                            else:
                                st.markdown(f"- {title}")

                # RAG sources
                sources = info.get("sources", [])
                if sources:
                    with st.expander(f"📄 {len(sources)} knowledge source(s) used"):
                        for s in sources:
                            st.markdown(f"- `{s}`")

                # Rewrite notice
                rewritten = info.get("rewritten_question")
                if rewritten:
                    st.caption(f"🔄 Query rewritten to: *{rewritten}*")

    # ── Force intent toggle (power user option) ────────────────────────────────
    with st.expander("⚙️ Advanced: force search mode", expanded=False):
        force_intent = st.selectbox(
            "Pin the planner to a specific mode (optional)",
            options=["Auto (recommended)", "rag_only", "web_search",
                     "university_recommender", "recommend_and_search"],
            key="force_intent_select",
        )

    # ── Chat input ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask me anything about studying in Germany…")
    if not user_input:
        return

    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Resolve forced intent (None = auto)
    resolved_intent = (
        None if force_intent == "Auto (recommended)" else force_intent
    )

    # ── Call backend ──────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            payload = {
                "question":        user_input,
                "user_id":         st.session_state.user_id,
                "conversation_id": st.session_state.conversation_id,
                "use_hybrid":      True,
                "use_window_expansion": True,
                "force_intent":    resolved_intent,
            }
            r = api("post", "/query", json=payload)

        if not r or not r.ok:
            st.error("Backend error. Please check that the server is running.")
            return

        data = r.json()

        # Display answer
        st.markdown(data["answer"])

        # Intent badge
        intent = data.get("intent")
        if intent:
            st.markdown(_intent_badge(intent), unsafe_allow_html=True)

        # Web sources
        web_sources = data.get("web_sources") or []
        if web_sources:
            with st.expander(f"🔍 {len(web_sources)} live web source(s) checked"):
                for src in web_sources:
                    title = src.get("title") or src.get("url", "")
                    url   = src.get("url", "")
                    if url:
                        st.markdown(f"- [{title}]({url})")

        # RAG sources
        sources = data.get("sources", [])
        if sources:
            with st.expander(f"📄 {len(sources)} knowledge source(s) used"):
                for s in sources:
                    st.markdown(f"- `{s}`")

        # Rewrite notice
        rewritten = data.get("rewritten_question")
        if rewritten:
            st.caption(f"🔄 Query rewritten to: *{rewritten}*")

    # Save to session
    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
    st.session_state.debug_info.append({
        "intent":             intent,
        "web_sources":        web_sources,
        "sources":            sources,
        "rewritten_question": rewritten,
    })


# ─────────────────────────────────────────────────────────────────────────────
# ④ UNIVERSITY FINDER PAGE (NEW v8)
# ─────────────────────────────────────────────────────────────────────────────

def show_university_finder():
    """
    Dedicated university recommendation page.
    Calls POST /recommend with the student's saved profile.
    """
    st.title("🏫 University Finder")
    st.caption(
        "Based on your profile, the hybrid recommender filters universities by your "
        "hard constraints (city, language, budget) and then ranks them by how well "
        "they match your academic interests."
    )

    profile = st.session_state.profile

    # ── Profile summary ────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Target Degree",   profile.get("target_degree", "Not set"))
    col2.metric("Field of Study",  profile.get("field_of_study", "Not set"))
    col3.metric("Monthly Budget",  f"€{profile.get('budget_monthly_eur', '?')}")

    st.info(
        f"**Preferred cities:** {profile.get('target_cities', 'Any')}  •  "
        f"**German level:** {profile.get('german_level', 'Not set')}  •  "
        f"**Target intake:** {profile.get('intake_semester', 'Not set')}"
    )

    # ── Optional free-text query to improve vector matching ───────────────────
    extra_query = st.text_input(
        "Describe what matters most to you (optional, improves matching):",
        placeholder="e.g. strong AI research, English-taught, near industry hub, good nightlife",
        key="uni_extra_query",
    )

    top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

    if st.button("🔍 Find Universities", type="primary"):
        with st.spinner("Searching and ranking universities…"):
            payload = {
                "profile":  profile,
                "question": extra_query,
                "top_k":    top_k,
            }
            r = api("post", "/recommend", json=payload)

        if not r or not r.ok:
            st.error("Could not reach the recommender. Is the backend running?")
            return

        data  = r.json()
        unis  = data.get("universities", [])
        total = data.get("total_filtered", 0)

        if not unis:
            st.warning(
                f"No universities matched your profile's hard constraints "
                f"(degree, field, city, language, budget). "
                f"Try relaxing your preferred cities or budget in your profile."
            )
            return

        st.success(f"Found **{total}** matching programs. Showing top {len(unis)}:")
        st.divider()

        for i, uni in enumerate(unis, 1):
            # ── University card ────────────────────────────────────────────────
            score   = uni.get("match_score", 0.5)
            pct     = int(score * 100)
            lang    = uni.get("language", "—")
            lang_icon = "🇬🇧" if lang == "English" else "🇩🇪" if lang == "German" else "🌍"

            with st.container(border=True):
                h_col, s_col = st.columns([3, 1])
                with h_col:
                    st.markdown(
                        f"### {i}. {uni['name']}  "
                        f"<span style='font-size:0.85rem; color:grey;'>"
                        f"{uni.get('city','')}, {uni.get('state','')}</span>",
                        unsafe_allow_html=True,
                    )
                    if uni.get("program_name"):
                        st.markdown(f"**Program:** {uni['program_name']}")

                with s_col:
                    # Match score progress bar
                    st.markdown(f"**Match: {pct}%**")
                    st.progress(score)

                # Key facts in columns
                f1, f2, f3, f4 = st.columns(4)
                f1.markdown(f"**Language**  \n{lang_icon} {lang}")
                f2.markdown(f"**Semester fee**  \n€{uni.get('semester_fee_eur', '?')}")
                f3.markdown(f"**German required**  \n{uni.get('german_required', '—')}")
                f4.markdown(f"**Min. GPA (DE)**  \n{uni.get('min_gpa', '—')}")

                # Deadlines
                d_col1, d_col2 = st.columns(2)
                if uni.get("winter_deadline"):
                    d_col1.markdown(f"📅 **Winter deadline:** {uni['winter_deadline']}")
                if uni.get("summer_deadline"):
                    d_col2.markdown(f"📅 **Summer deadline:** {uni['summer_deadline']}")

                # Research areas
                if uni.get("research_areas"):
                    st.markdown(f"🔬 **Research areas:** {uni['research_areas']}")

                # Strengths
                if uni.get("strengths"):
                    st.markdown(f"⭐ {uni['strengths']}")

                # Match reason
                if uni.get("match_reason"):
                    st.caption(f"💡 Why matched: {uni['match_reason']}")

                # Official link
                if uni.get("url"):
                    st.markdown(f"[🔗 Official program page]({uni['url']})")

            st.write("")  # vertical spacing between cards


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ PROFILE EDITOR PAGE
# ─────────────────────────────────────────────────────────────────────────────

def show_profile_editor():
    """Let the student update their profile at any time."""
    st.title("👤 Edit Your Profile")
    st.caption("Updating your profile improves all recommendations and advice.")

    profile = st.session_state.profile

    with st.form("edit_profile_form"):
        new_profile: dict = {}

        for field in PROFILE_FIELDS:
            key     = field["key"]
            label   = field["label"]
            help_   = field.get("help", "")
            current = profile.get(key, field.get("default", ""))

            if field["type"] == "text":
                new_profile[key] = st.text_input(label, value=str(current), help=help_)
            elif field["type"] == "select":
                options = field["options"]
                idx     = options.index(current) if current in options else 0
                new_profile[key] = st.selectbox(label, options=options, index=idx, help=help_)
            elif field["type"] == "number":
                new_profile[key] = st.number_input(label, value=int(current or 0),
                                                    min_value=0, help=help_)
            elif field["type"] == "textarea":
                new_profile[key] = st.text_area(label, value=str(current), help=help_)

        saved = st.form_submit_button("💾 Save profile")

    if saved:
        r = api("put", f"/users/{st.session_state.user_id}/profile",
                json={"profile": new_profile})
        if r and r.ok:
            st.session_state.profile = new_profile
            st.success("✅ Profile updated!")
        else:
            st.error("Could not save profile.")


# ─────────────────────────────────────────────────────────────────────────────
# App entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # If no user is set up, show the onboarding screen
    if not st.session_state.user_id:
        show_profile_setup()
        return

    # Render sidebar for logged-in users
    render_sidebar()

    # Route to the active page
    page = st.session_state.active_page
    if page == "chat":
        show_chat_page()
    elif page == "universities":
        show_university_finder()
    elif page == "profile":
        show_profile_editor()
    else:
        show_chat_page()


if __name__ == "__main__":
    main()