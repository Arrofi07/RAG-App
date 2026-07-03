# streamlit_app.py  (v10.0.0)
#
# V10 CHANGES
# ───────────
# 1. AUTH PAGES — Login and Register screens shown when no JWT is in session.
#    JWT stored in st.session_state["token"]. Lost on hard page refresh
#    (known limitation of Streamlit — see auth.py for the design note).
#
# 2. BEARER TOKEN on every api() call — the Authorization header is injected
#    automatically so callers don't need to think about it.
#
# 3. EVAL DASHBOARD PAGE — new sidebar nav item. Shows historical run scores
#    as a line chart, lets you generate questions and trigger eval runs.
#
# 4. DUPLICATE HANDLING — the ingest form now shows a clear message on 409
#    (duplicate) instead of a generic "Ingestion failed" error.
#
# 5. LOGOUT button in the sidebar.

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from advisor_config import PROFILE_FIELDS

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

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
    "token":           None,    # JWT string — presence means logged in
    "user_id":         None,
    "user_name":       None,
    "user_email":      None,
    "profile":         {},
    "conversation_id": None,
    "messages":        [],
    "debug_info":      [],
    "active_page":     "chat",  # "chat" | "universities" | "profile" | "eval"
    "auth_tab":        "login", # "login" | "register"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# API helper — injects JWT automatically on every call
# ─────────────────────────────────────────────────────────────────────────────

def api(method: str, path: str, timeout: int = 60, **kwargs):
    """
    Make an authenticated API call. Injects the Bearer token automatically.

    Returns the Response object or None on network/timeout error.
    Handles 401 globally — clears session and forces re-login.
    """
    url     = f"{API_BASE}{path}"
    headers = kwargs.pop("headers", {})

    # Inject JWT if we have one
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"

    try:
        resp = getattr(requests, method)(url, timeout=timeout, headers=headers, **kwargs)

        # Global 401 handler — token expired or invalid
        if resp.status_code == 401:
            st.session_state.token    = None
            st.session_state.user_id  = None
            st.session_state.user_name = None
            st.warning("Your session has expired. Please log in again.")
            st.rerun()

        return resp

    except requests.exceptions.ReadTimeout:
        st.error(
            f"⏱️ Request timed out after {timeout}s. "
            "For large PDF ingestion this is normal — check your backend terminal. "
            "Ingestion may still be completing in the background."
        )
        return None
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None


def _auth_headers() -> dict:
    """Return Authorization header dict for use in api() calls that need explicit headers."""
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}


@st.cache_data(ttl=30)
def fetch_doc_metadata(_token: str):
    """Cached fetch of document metadata. _token arg busts cache on login."""
    r = api("get", "/documents/metadata")
    return r.json().get("documents", []) if r and r.ok else []


def load_conversation(conv_id: str):
    """Load persisted messages from DB into session state."""
    r = api("get", f"/conversations/{conv_id}/messages")
    if r and r.ok:
        msgs = r.json().get("messages", [])
        st.session_state.messages   = [{"role": m["role"], "content": m["content"]} for m in msgs]
        assistant_count             = sum(1 for m in msgs if m["role"] == "assistant")
        st.session_state.debug_info = [{} for _ in range(assistant_count)]


# ─────────────────────────────────────────────────────────────────────────────
# Intent badge helper (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_LABELS = {
    "rag_only":               ("📄", "Knowledge Base",    "#3b82f6"),
    "web_search":             ("🔍", "Live Web Search",   "#10b981"),
    "university_recommender": ("🏫", "University Finder", "#8b5cf6"),
    "recommend_and_search":   ("🔍🏫", "Finder + Web",    "#f59e0b"),
}

def _intent_badge(intent: str) -> str:
    emoji, label, colour = _INTENT_LABELS.get(intent, ("⚙️", intent, "#6b7280"))
    return (
        f'<span style="background:{colour}22; color:{colour}; border:1px solid {colour}; '
        f'border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600;">'
        f'{emoji} {label}</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# ① AUTH PAGES — Login and Register
# ─────────────────────────────────────────────────────────────────────────────

def show_auth_page():
    """
    Full-page login / register flow shown when no JWT is in session state.

    Design note: we don't use st.tabs() here because switching tabs re-runs
    the whole script and clears form state. Instead we use buttons to set
    st.session_state.auth_tab and rerun.
    """
    st.title("🎓 Study in Germany — AI Advisor")

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        # Tab switcher
        tab_col1, tab_col2 = st.columns(2)
        if tab_col1.button(
            "**🔑 Log In**" if st.session_state.auth_tab == "login" else "🔑 Log In",
            use_container_width=True,
            type="primary" if st.session_state.auth_tab == "login" else "secondary",
        ):
            st.session_state.auth_tab = "login"
            st.rerun()

        if tab_col2.button(
            "**📝 Register**" if st.session_state.auth_tab == "register" else "📝 Register",
            use_container_width=True,
            type="primary" if st.session_state.auth_tab == "register" else "secondary",
        ):
            st.session_state.auth_tab = "register"
            st.rerun()

        st.divider()

        if st.session_state.auth_tab == "login":
            _show_login_form()
        else:
            _show_register_form()


def _show_login_form():
    st.subheader("Welcome back")

    with st.form("login_form", clear_on_submit=False):
        email    = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submit   = st.form_submit_button("Log In", use_container_width=True, type="primary")

    if submit:
        if not email or not password:
            st.error("Please enter your email and password.")
            return

        r = api("post", "/auth/login", json={"email": email, "password": password})

        if r and r.ok:
            data = r.json()
            _set_session(data)
            st.success(f"Welcome back, {data['name']}!")
            st.rerun()
        elif r is not None:
            try:
                detail = r.json().get("detail", "Login failed.")
            except Exception:
                detail = "Login failed."
            st.error(detail)


def _show_register_form():
    st.subheader("Create your account")
    st.caption("Already have an account? Click **Log In** above.")

    with st.form("register_form", clear_on_submit=False):
        name     = st.text_input("Full name *", placeholder="e.g. Budi Santoso")
        email    = st.text_input("Email *",     placeholder="you@example.com")
        password = st.text_input("Password *",  type="password",
                                  help="At least 8 characters, must include a number or uppercase letter.")
        password2 = st.text_input("Confirm password *", type="password")

        st.divider()
        st.caption("**Your study profile** — fill in what you know, update later anytime")

        profile_data: dict = {}
        for field in PROFILE_FIELDS:
            key   = field["key"]
            label = field["label"]
            help_ = field.get("help", "")
            if field["type"] == "text":
                profile_data[key] = st.text_input(label, help=help_, value=field.get("default", ""))
            elif field["type"] == "select":
                profile_data[key] = st.selectbox(
                    label, options=field["options"], help=help_,
                    index=field["options"].index(field["default"])
                    if field.get("default") in field["options"] else 0,
                )
            elif field["type"] == "number":
                profile_data[key] = st.number_input(label, help=help_,
                                                     value=field.get("default", 0), min_value=0)
            elif field["type"] == "textarea":
                profile_data[key] = st.text_area(label, help=help_, value=field.get("default", ""))

        submit = st.form_submit_button("Create Account & Start", use_container_width=True, type="primary")

    if submit:
        # Client-side validation before hitting the API
        errors = []
        if not name.strip():
            errors.append("Name is required.")
        if not email.strip() or "@" not in email:
            errors.append("A valid email is required.")
        if not password:
            errors.append("Password is required.")
        elif password != password2:
            errors.append("Passwords do not match.")
        elif len(password) < 8:
            errors.append("Password must be at least 8 characters.")

        if errors:
            for e in errors:
                st.error(e)
            return

        r = api("post", "/auth/register", json={
            "name":     name.strip(),
            "email":    email.strip(),
            "password": password,
            "profile":  profile_data,
        })

        if r and r.ok:
            data = r.json()
            _set_session(data)

            # Create first conversation
            rc = api("post", f"/users/{data['user_id']}/conversations",
                     params={"title": "First conversation"})
            if rc and rc.ok:
                st.session_state.conversation_id = rc.json()["conversation_id"]

            st.success(f"Account created! Welcome, {data['name']}.")
            st.rerun()
        elif r is not None:
            try:
                detail = r.json().get("detail", "Registration failed.")
            except Exception:
                detail = "Registration failed."
            # 409 = email already registered
            if r.status_code == 409:
                st.error(f"{detail} Try logging in instead.")
            else:
                st.error(detail)


def _set_session(auth_data: dict):
    """Populate session state from a successful auth response."""
    st.session_state.token      = auth_data["access_token"]
    st.session_state.user_id    = auth_data["user_id"]
    st.session_state.user_name  = auth_data["name"]
    st.session_state.user_email = auth_data["email"]

    # Fetch full profile
    r = api("get", "/auth/me")
    if r and r.ok:
        st.session_state.profile = r.json().get("profile", {})


# ─────────────────────────────────────────────────────────────────────────────
# ② SIDEBAR (shown when logged in)
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user_name}")
        st.caption(st.session_state.user_email or "")

        # ── Page navigation ────────────────────────────────────────────────────
        st.divider()
        pages = [
            ("chat",         "💬 Chat with Advisor"),
            ("universities", "🏫 Find Universities"),
            ("eval",         "📊 Eval Dashboard"),
            ("profile",      "👤 Edit My Profile"),
        ]
        for page_id, label in pages:
            btn_type = "primary" if st.session_state.active_page == page_id else "secondary"
            if st.button(label, use_container_width=True, type=btn_type, key=f"nav_{page_id}"):
                st.session_state.active_page = page_id
                st.rerun()

        # ── Conversation management ────────────────────────────────────────────
        st.divider()
        st.markdown("**Conversations**")

        r    = api("get", f"/users/{st.session_state.user_id}/conversations")
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

        # ── Document upload ────────────────────────────────────────────────────
        st.divider()
        st.markdown("**📄 Upload Knowledge Documents**")
        with st.expander("Upload PDF"):
            _render_upload_form()

        # ── Logout ─────────────────────────────────────────────────────────────
        st.divider()
        if st.button("🚪 Log Out", use_container_width=True):
            for key in ["token", "user_id", "user_name", "user_email", "profile",
                        "conversation_id", "messages", "debug_info"]:
                st.session_state[key] = None if key == "token" else (
                    {} if key == "profile" else ([] if key in ("messages", "debug_info") else None)
                )
            st.session_state.active_page = "chat"
            st.rerun()

        st.caption("v10.0.0 — Auth + Dedup + Eval")


def _render_upload_form():
    """PDF upload form with deduplication-aware error handling and time estimate."""
    uploaded = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")
    if not uploaded:
        return

    category = st.text_input("Category (optional)", key="cat_inp")
    author   = st.text_input("Author (optional)",   key="auth_inp")
    year     = st.number_input("Year (optional)", min_value=2000, max_value=2030,
                                value=2024, key="year_inp")
    tags     = st.text_input("Tags (comma-separated)", key="tags_inp")
    use_ctx  = st.checkbox("Contextual enrichment (slower, smarter)", value=False)

    # Time estimate
    est_pages, est_seconds = None, None
    try:
        import pypdf
        reader    = pypdf.PdfReader(uploaded)
        est_pages = len(reader.pages)
        uploaded.seek(0)
    except Exception:
        pass

    if use_ctx and est_pages:
        is_local      = os.getenv("ENRICHMENT_PROVIDER", "ollama").lower() == "ollama"
        batch_size    = int(os.getenv("CONTEXT_BATCH_SIZE", "20" if is_local else "5"))
        delay         = float(os.getenv("CONTEXT_INTER_BATCH_DELAY", "0" if is_local else "5"))
        est_batches   = max(1, -(-est_pages // batch_size))
        est_seconds   = int(est_batches * (delay + (3 if is_local else 5)))
        time_str      = (f"{est_seconds // 60}m {est_seconds % 60}s"
                         if est_seconds >= 60 else f"~{est_seconds}s")
        st.info(
            f"📄 **{est_pages} pages** → {est_batches} batches → **{time_str}** "
            + ("(local, no pauses) ✅" if is_local else "(cloud, includes pauses)")
        )
    elif use_ctx:
        st.info("Contextual enrichment on. Large PDFs take a few minutes.")

    if st.button("⬆️ Ingest PDF"):
        dynamic_timeout = (
            min(max((est_seconds or 0) + 60, 120), 1200)
            if use_ctx and est_seconds else 300
        )

        placeholder = st.empty()
        placeholder.info(
            f"⏳ Ingesting… waiting up to {dynamic_timeout // 60}m. "
            "Watch backend terminal for live progress."
        )

        with st.spinner("Ingesting…"):
            r = api(
                "post", "/ingest",
                timeout=dynamic_timeout,
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                data={
                    "category":       category or "",
                    "author":         author or "",
                    "year":           str(year),
                    "tags":           tags or "",
                    "use_contextual": "true" if use_ctx else "false",
                },
            )

        placeholder.empty()

        if r and r.ok:
            res = r.json()
            st.success(f"✅ {res['chunks']} chunks ingested from **{res['source']}**")
            st.cache_data.clear()
        elif r is not None:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text

            # 409 = duplicate — give a specific, helpful message
            if r.status_code == 409:
                st.warning(f"⚠️ Duplicate detected: {detail}")
            else:
                st.error(f"Ingestion failed ({r.status_code}): {detail}")
        else:
            # Timeout — offer a recovery check
            if st.button("🔄 Check if ingestion finished"):
                check = api("get", "/documents/metadata", timeout=15)
                if check and check.ok:
                    docs  = check.json().get("documents", [])
                    names = [d.get("filename", "") for d in docs]
                    if uploaded.name in names:
                        st.success(
                            f"✅ '{uploaded.name}' IS in the knowledge base. "
                            "The ingestion succeeded — only the frontend connection timed out."
                        )
                        st.cache_data.clear()
                    else:
                        st.warning("Not found yet. Check backend terminal and try again in a minute.")


# ─────────────────────────────────────────────────────────────────────────────
# ③ CHAT PAGE (unchanged logic, auth now in api())
# ─────────────────────────────────────────────────────────────────────────────

def show_chat_page():
    st.title("💬 AI Advisor Chat")

    if not st.session_state.conversation_id:
        rc = api("post", f"/users/{st.session_state.user_id}/conversations")
        if rc and rc.ok:
            st.session_state.conversation_id = rc.json()["conversation_id"]

    debug_idx = 0
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and debug_idx < len(st.session_state.debug_info):
                info = st.session_state.debug_info[debug_idx]
                debug_idx += 1

                if info.get("intent"):
                    st.markdown(_intent_badge(info["intent"]), unsafe_allow_html=True)

                web_sources = info.get("web_sources", [])
                if web_sources:
                    with st.expander(f"🔍 {len(web_sources)} live web source(s)"):
                        for src in web_sources:
                            title = src.get("title") or src.get("url", "")
                            url   = src.get("url", "")
                            st.markdown(f"- [{title}]({url})" if url else f"- {title}")

                if info.get("sources"):
                    with st.expander(f"📄 {len(info['sources'])} knowledge source(s)"):
                        for s in info["sources"]:
                            st.markdown(f"- `{s}`")

                if info.get("rewritten_question"):
                    st.caption(f"🔄 Query rewritten: *{info['rewritten_question']}*")

    with st.expander("⚙️ Advanced: force search mode", expanded=False):
        force_intent = st.selectbox(
            "Pin the planner to a specific mode (optional)",
            ["Auto (recommended)", "rag_only", "web_search",
             "university_recommender", "recommend_and_search"],
            key="force_intent_select",
        )

    user_input = st.chat_input("Ask me anything about studying in Germany…")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    resolved_intent = None if force_intent == "Auto (recommended)" else force_intent

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            r = api("post", "/query", json={
                "question":             user_input,
                "user_id":              st.session_state.user_id,
                "conversation_id":      st.session_state.conversation_id,
                "use_hybrid":           True,
                "use_window_expansion": True,
                "force_intent":         resolved_intent,
            })

        if not r or not r.ok:
            st.error("Backend error. Is the server running?")
            return

        data   = r.json()
        intent = data.get("intent")

        st.markdown(data["answer"])
        if intent:
            st.markdown(_intent_badge(intent), unsafe_allow_html=True)

        web_sources = data.get("web_sources") or []
        if web_sources:
            with st.expander(f"🔍 {len(web_sources)} live web source(s)"):
                for src in web_sources:
                    url = src.get("url", "")
                    st.markdown(f"- [{src.get('title', url)}]({url})" if url else f"- {src.get('title', '')}")

        sources = data.get("sources", [])
        if sources:
            with st.expander(f"📄 {len(sources)} knowledge source(s)"):
                for s in sources:
                    st.markdown(f"- `{s}`")

        if data.get("rewritten_question"):
            st.caption(f"🔄 Query rewritten: *{data['rewritten_question']}*")

    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
    st.session_state.debug_info.append({
        "intent":             intent,
        "web_sources":        web_sources,
        "sources":            sources,
        "rewritten_question": data.get("rewritten_question"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# ④ UNIVERSITY FINDER PAGE (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def show_university_finder():
    st.title("🏫 University Finder")
    st.caption(
        "SQL hard-filter on your constraints → vector soft-match on your interests → ranked results."
    )

    profile = st.session_state.profile
    c1, c2, c3 = st.columns(3)
    c1.metric("Target Degree",  profile.get("target_degree",  "Not set"))
    c2.metric("Field of Study", profile.get("field_of_study", "Not set"))
    c3.metric("Monthly Budget", f"€{profile.get('budget_monthly_eur', '?')}")

    st.info(
        f"**Cities:** {profile.get('target_cities', 'Any')}  •  "
        f"**German:** {profile.get('german_level', 'Not set')}  •  "
        f"**Intake:** {profile.get('intake_semester', 'Not set')}"
    )

    extra_query = st.text_input(
        "Describe what matters most (optional — improves ranking):",
        placeholder="e.g. strong AI research, English-taught, good industry connections",
        key="uni_extra_query",
    )
    top_k = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("🔍 Find Universities", type="primary"):
        with st.spinner("Filtering and ranking…"):
            r = api("post", "/recommend", json={
                "profile":  profile,
                "question": extra_query,
                "top_k":    top_k,
            })

        if not r or not r.ok:
            st.error("Could not reach the recommender.")
            return

        data  = r.json()
        unis  = data.get("universities", [])
        total = data.get("total_filtered", 0)

        if not unis:
            st.warning(
                "No universities matched your hard constraints. "
                "Try relaxing city, budget, or language requirements in your profile."
            )
            return

        st.success(f"**{total}** programs passed hard filters. Showing top {len(unis)}:")
        st.divider()

        for i, uni in enumerate(unis, 1):
            score     = uni.get("match_score", 0.5)
            lang      = uni.get("language", "—")
            lang_icon = "🇬🇧" if lang == "English" else "🇩🇪" if lang == "German" else "🌍"

            with st.container(border=True):
                h_col, s_col = st.columns([3, 1])
                with h_col:
                    st.markdown(
                        f"### {i}. {uni['name']} "
                        f"<span style='font-size:0.85rem;color:grey'>"
                        f"{uni.get('city','')}, {uni.get('state','')}</span>",
                        unsafe_allow_html=True,
                    )
                    if uni.get("program_name"):
                        st.markdown(f"**Program:** {uni['program_name']}")
                with s_col:
                    st.markdown(f"**Match: {int(score*100)}%**")
                    st.progress(score)

                f1, f2, f3, f4 = st.columns(4)
                f1.markdown(f"**Language**\n{lang_icon} {lang}")
                f2.markdown(f"**Semester fee**\n€{uni.get('semester_fee_eur','?')}")
                f3.markdown(f"**German req.**\n{uni.get('german_required','—')}")
                f4.markdown(f"**Min GPA**\n{uni.get('min_gpa','—')}")

                d1, d2 = st.columns(2)
                if uni.get("winter_deadline"):
                    d1.markdown(f"📅 Winter: **{uni['winter_deadline']}**")
                if uni.get("summer_deadline"):
                    d2.markdown(f"📅 Summer: **{uni['summer_deadline']}**")

                if uni.get("research_areas"):
                    st.markdown(f"🔬 {uni['research_areas']}")
                if uni.get("strengths"):
                    st.markdown(f"⭐ {uni['strengths']}")
                if uni.get("match_reason"):
                    st.caption(f"💡 {uni['match_reason']}")
                if uni.get("url"):
                    st.markdown(f"[🔗 Official page]({uni['url']})")

            st.write("")


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ EVAL DASHBOARD PAGE (new in v10)
# ─────────────────────────────────────────────────────────────────────────────

def show_eval_page():
    """
    Evaluation dashboard with two sections:
      A. Question management — generate / view / delete the synthetic eval dataset
      B. Run history — line chart of Hit Rate and MRR over time + on-demand triggers
    """
    st.title("📊 Evaluation Dashboard")
    st.caption(
        "Track retrieval quality over time. "
        "Retrieval metrics (Hit Rate, MRR) are free and fast. "
        "Answer quality (RAGAS faithfulness + relevancy) costs Gemini quota — run on-demand."
    )

    # ── Section A: Question dataset ────────────────────────────────────────────
    st.subheader("A. Eval Question Dataset")

    r_q = api("get", "/eval/questions")
    questions = r_q.json().get("questions", []) if r_q and r_q.ok else []
    st.metric("Stored eval questions", len(questions))

    col_gen, col_del = st.columns(2)

    with col_gen:
        n_gen = st.number_input("Questions to generate", min_value=5, max_value=100,
                                value=20, step=5, key="n_gen")
        if st.button("⚙️ Generate Questions", type="primary", use_container_width=True):
            with st.spinner(
                f"Sampling {n_gen} chunks and generating questions via local Qwen… "
                "This may take a minute."
            ):
                r = api("post", f"/eval/generate-questions?n={n_gen}", timeout=300)
            if r and r.ok:
                data = r.json()
                st.success(f"✅ Generated {data['generated']} questions.")
                st.rerun()
            elif r:
                st.error(r.json().get("detail", "Generation failed."))

    with col_del:
        st.write("")
        st.write("")
        if st.button("🗑 Delete All Questions", use_container_width=True):
            r = api("delete", "/eval/questions")
            if r and r.ok:
                st.success(f"Deleted {r.json()['deleted']} questions.")
                st.rerun()

    if questions:
        with st.expander(f"Preview {min(5, len(questions))} questions"):
            for q in questions[:5]:
                st.markdown(f"- **{q['question']}**  \n  ↳ source: `{q['source_doc']}`")

    st.divider()

    # ── Section B: Run history and triggers ────────────────────────────────────
    st.subheader("B. Eval Runs")

    r_runs = api("get", "/eval/runs")
    runs   = r_runs.json().get("runs", []) if r_runs and r_runs.ok else []

    # ── Metric chart ───────────────────────────────────────────────────────────
    if runs:
        import pandas as pd

        retrieval_runs = [r for r in runs if r["run_type"] == "retrieval" and r.get("hit_rate") is not None]
        aq_runs        = [r for r in runs if r["run_type"] == "answer_quality"]

        if retrieval_runs:
            st.markdown("**Retrieval metrics over time**")
            df = pd.DataFrame([{
                "Date":      r["ran_at"][:10],
                "Hit Rate":  round(r["hit_rate"], 3),
                "MRR":       round(r.get("mrr") or 0, 3),
                "Questions": r["num_questions"],
            } for r in retrieval_runs])
            st.line_chart(df.set_index("Date")[["Hit Rate", "MRR"]])
            st.dataframe(df, use_container_width=True, hide_index=True)

        if aq_runs:
            st.markdown("**Answer quality metrics over time**")
            df_aq = pd.DataFrame([{
                "Date":             r["ran_at"][:10],
                "Faithfulness":     round(r.get("faithfulness") or 0, 3),
                "Answer Relevancy": round(r.get("answer_relevancy") or 0, 3),
                "Questions":        r["num_questions"],
            } for r in aq_runs])
            st.line_chart(df_aq.set_index("Date")[["Faithfulness", "Answer Relevancy"]])
            st.dataframe(df_aq, use_container_width=True, hide_index=True)
    else:
        st.info("No eval runs yet. Generate questions and run an evaluation below.")

    st.divider()

    # ── Run triggers ───────────────────────────────────────────────────────────
    st.subheader("Run Evaluation")

    trig_col1, trig_col2 = st.columns(2)

    with trig_col1:
        with st.container(border=True):
            st.markdown("**🚀 Retrieval Eval**")
            st.caption("Hit Rate + MRR. Fast, free, no LLM calls.")
            top_k_r = st.slider("Top-K", 1, 10, 5, key="ret_topk")
            notes_r = st.text_input("Notes (optional)", key="ret_notes",
                                    placeholder="e.g. after adding 5 new PDFs")
            if st.button("Run Retrieval Eval", type="primary", use_container_width=True):
                if not questions:
                    st.warning("Generate eval questions first (Section A above).")
                else:
                    with st.spinner(f"Running retrieval eval on {len(questions)} questions…"):
                        r = api("post",
                                f"/eval/run-retrieval?top_k={top_k_r}&notes={notes_r}",
                                timeout=600)
                    if r and r.ok:
                        res = r.json()
                        st.success(
                            f"✅ Done — Hit Rate: **{res['hit_rate']:.1%}** | "
                            f"MRR: **{res['mrr']:.3f}** (n={res['n']})"
                        )
                        st.rerun()
                    elif r:
                        st.error(r.json().get("detail", "Eval failed."))

    with trig_col2:
        with st.container(border=True):
            st.markdown("**🧪 Answer Quality Eval** *(on-demand)*")
            st.caption("RAGAS faithfulness + relevancy. Costs Gemini quota.")
            n_aq    = st.number_input("Questions to sample", 3, 20, 10, key="aq_n")
            top_k_a = st.slider("Top-K", 1, 10, 5, key="aq_topk")
            notes_a = st.text_input("Notes (optional)", key="aq_notes")
            if st.button("Run Answer Quality Eval", use_container_width=True):
                if not questions:
                    st.warning("Generate eval questions first (Section A above).")
                else:
                    with st.spinner(
                        f"Running RAGAS eval on {n_aq} questions… "
                        f"~{n_aq * 7} Gemini calls, may take several minutes."
                    ):
                        r = api("post",
                                f"/eval/run-answer-quality?n_questions={n_aq}"
                                f"&top_k={top_k_a}&notes={notes_a}",
                                timeout=900)
                    if r and r.ok:
                        res = r.json()
                        faith   = res.get("faithfulness")
                        relev   = res.get("answer_relevancy")
                        st.success(
                            f"✅ Done — Faithfulness: **{faith:.1%}** | "
                            f"Relevancy: **{relev:.1%}** (n={res['n']})"
                            if faith and relev else
                            f"✅ Done (n={res['n']}) — some metrics may be None if LLM calls failed."
                        )
                        st.rerun()
                    elif r:
                        st.error(r.json().get("detail", "Eval failed."))

    # ── Drill-down: per-question results ───────────────────────────────────────
    if runs:
        st.divider()
        st.subheader("Drill-down: per-question results")
        run_options = {
            f"{r['ran_at'][:16]} | {r['run_type']} | n={r['num_questions']}": r["id"]
            for r in runs
        }
        selected_label = st.selectbox("Select a run", options=list(run_options.keys()))
        selected_run_id = run_options[selected_label]

        if st.button("Load results"):
            r_det = api("get", f"/eval/runs/{selected_run_id}", timeout=30)
            if r_det and r_det.ok:
                import pandas as pd
                results = r_det.json().get("results", [])
                if results:
                    df_det = pd.DataFrame([{
                        "Question":    res["question"][:80],
                        "Hit":         "✅" if res.get("hit") else "❌",
                        "Rank":        res.get("rank") or "—",
                        "Faithfulness": f"{res['faithfulness']:.2f}" if res.get("faithfulness") else "—",
                        "Relevancy":   f"{res['answer_relevancy']:.2f}" if res.get("answer_relevancy") else "—",
                    } for res in results])
                    st.dataframe(df_det, use_container_width=True, hide_index=True)
                else:
                    st.info("No per-question results found for this run.")


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ PROFILE EDITOR PAGE (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

def show_profile_editor():
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

        if st.form_submit_button("💾 Save profile"):
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
    # Show auth page if not logged in
    if not st.session_state.token:
        show_auth_page()
        return

    render_sidebar()

    page = st.session_state.active_page
    if page == "chat":
        show_chat_page()
    elif page == "universities":
        show_university_finder()
    elif page == "eval":
        show_eval_page()
    elif page == "profile":
        show_profile_editor()
    else:
        show_chat_page()


if __name__ == "__main__":
    main()