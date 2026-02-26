import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
ROOT_DIR = Path(__file__).resolve().parent
RUN_API_SCRIPT = ROOT_DIR / "scripts" / "run_api.ps1"


def api_reachable(timeout: int = 3) -> bool:
    try:
        res = requests.get(f"{API_URL}/health", timeout=timeout)
        return res.status_code == 200
    except Exception:
        return False


def start_backend() -> bool:
    if not RUN_API_SCRIPT.exists():
        return False
    subprocess.Popen(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(RUN_API_SCRIPT)],
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def wait_for_backend(max_wait_seconds: int = 75, interval_seconds: int = 3) -> bool:
    waited = 0
    while waited < max_wait_seconds:
        if api_reachable(timeout=2):
            return True
        time.sleep(interval_seconds)
        waited += interval_seconds
    return False


def auth_headers() -> dict[str, str]:
    token = st.session_state.get("access_token", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_get(path: str, params: dict | None = None, auth: bool = False, timeout: int = 20) -> requests.Response:
    headers = auth_headers() if auth else None
    return requests.get(f"{API_URL}{path}", params=params, headers=headers, timeout=timeout)


def api_post(path: str, json_payload: dict | None = None, auth: bool = False, timeout: int = 30) -> requests.Response:
    headers = auth_headers() if auth else None
    return requests.post(f"{API_URL}{path}", json=json_payload, headers=headers, timeout=timeout)


def format_answer_blocks(answer: str) -> str:
    formatted = answer
    labels = [
        "Short Answer:",
        "Topic Examples:",
        "Short Summary:",
        "Detailed Explanation:",
        "Real-Life Example:",
        "Key Points:",
        "Conclusion:",
    ]
    for label in labels:
        formatted = formatted.replace(label, f"\n\n### {label}\n")
    return formatted.strip()


def do_logout() -> None:
    st.session_state.is_authenticated = False
    st.session_state.access_token = ""
    st.session_state.user_id = ""
    st.session_state.role = ""
    st.session_state.messages = []
    st.session_state.last_chat_id = None
    st.session_state.last_improved = False
    st.session_state.last_variant = ""


def fetch_documents() -> dict:
    try:
        res = api_get("/documents", auth=True)
        res.raise_for_status()
        return res.json()
    except Exception:
        return {"total_pdfs": 0, "total_chunks": 0, "documents": []}


def rebuild_index() -> tuple[bool, str]:
    try:
        res = api_post("/rebuild-index", auth=True, timeout=3600)
        res.raise_for_status()
        data = res.json()
        return True, f"Indexed {data.get('total_pdfs', 0)} PDFs, {data.get('total_chunks', 0)} chunks"
    except Exception as exc:
        return False, str(exc)


def fetch_users(limit: int = 1000) -> list[dict]:
    try:
        res = api_get("/users", params={"limit": limit}, auth=True)
        res.raise_for_status()
        return res.json().get("users", [])
    except Exception:
        return []


def fetch_analytics(scope: str, user_id: str | None, weak_topics_limit: int = 5000) -> dict:
    try:
        params = {"scope": scope, "weak_topics_limit": weak_topics_limit}
        if user_id:
            params["user_id"] = user_id
        res = api_get("/analytics", params=params, auth=True)
        res.raise_for_status()
        return res.json()
    except Exception:
        return {
            "total_chats": 0,
            "total_feedback": 0,
            "avg_rating": 0,
            "avg_reward": 0,
            "weak_topics": [],
        }


def fetch_reward_history(user_id: str | None = None, limit: int = 2000) -> list[dict]:
    try:
        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        res = api_get("/reward-history", params=params, auth=True)
        res.raise_for_status()
        return res.json().get("history", [])
    except Exception:
        return []


st.set_page_config(page_title="Academic Intelligence System", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    .stTextInput input, .stTextArea textarea { color: #e6edf3 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("A Reinforcement-Aligned Academic Intelligence System")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_chat_id" not in st.session_state:
    st.session_state.last_chat_id = None
if "last_mode" not in st.session_state:
    st.session_state.last_mode = "detailed"
if "autostart_attempted" not in st.session_state:
    st.session_state.autostart_attempted = False
if "last_improved" not in st.session_state:
    st.session_state.last_improved = False
if "last_variant" not in st.session_state:
    st.session_state.last_variant = ""
if "is_authenticated" not in st.session_state:
    st.session_state.is_authenticated = False
if "access_token" not in st.session_state:
    st.session_state.access_token = ""
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "role" not in st.session_state:
    st.session_state.role = ""


def render_login_page(backend_available: bool) -> None:
    left, center, right = st.columns([1.2, 1.0, 1.2])
    with center:
        st.markdown("### Login")
        st.caption("Enter credentials to access the system")

        card = st.container(border=True)
        with card:
            login_tab, register_tab = st.tabs(["Login", "Register"])

            with login_tab:
                with st.form("login_form"):
                    login_uid = st.text_input("User ID")
                    login_pwd = st.text_input("Password", type="password")
                    st.checkbox("Remember me", value=True)
                    login_submit = st.form_submit_button("Login", use_container_width=True, disabled=not backend_available)

                if login_submit:
                    try:
                        res = api_post(
                            "/auth/login",
                            json_payload={"user_id": login_uid.strip(), "password": login_pwd},
                            timeout=30,
                        )
                        if res.status_code != 200:
                            detail = "Invalid credentials."
                            try:
                                detail = res.json().get("detail", detail)
                            except Exception:
                                pass
                            st.error(detail)
                        else:
                            data = res.json()
                            st.session_state.is_authenticated = True
                            st.session_state.access_token = data["access_token"]
                            st.session_state.user_id = data["user_id"]
                            st.session_state.role = data["role"]
                            st.session_state.messages = []
                            st.success("Login successful.")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Login failed: {exc}")

            with register_tab:
                with st.form("register_form"):
                    reg_uid = st.text_input("New User ID")
                    reg_pwd = st.text_input("New Password", type="password")
                    reg_pwd2 = st.text_input("Confirm Password", type="password")
                    reg_submit = st.form_submit_button("Create User", use_container_width=True, disabled=not backend_available)

                if reg_submit:
                    if reg_pwd != reg_pwd2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            res = api_post(
                                "/auth/register",
                                json_payload={"user_id": reg_uid.strip(), "password": reg_pwd},
                                timeout=30,
                            )
                            if res.status_code != 200:
                                detail = "Registration failed."
                                try:
                                    detail = res.json().get("detail", detail)
                                except Exception:
                                    pass
                                st.error(detail)
                            else:
                                data = res.json()
                                st.session_state.is_authenticated = True
                                st.session_state.access_token = data["access_token"]
                                st.session_state.user_id = data["user_id"]
                                st.session_state.role = data["role"]
                                st.session_state.messages = []
                                st.success("Account created and logged in.")
                                st.rerun()
                        except Exception as exc:
                            st.error(f"Registration failed: {exc}")


backend_up = api_reachable()
if not backend_up and not st.session_state.autostart_attempted:
    st.session_state.autostart_attempted = True
    if start_backend():
        backend_up = wait_for_backend()

short_answer = False
detailed_explanation = True
show_sources = True
admin_scope = "all"
admin_user_filter = ""

with st.sidebar:
    st.caption(f"API Endpoint: {API_URL}")
    if backend_up:
        st.success("Backend: Connected")
    else:
        st.error("Backend: Not running")
        if st.button("Start Backend", use_container_width=True):
            if start_backend():
                wait_for_backend()
                st.rerun()
            else:
                st.error("Unable to find scripts/run_api.ps1")

    if st.session_state.is_authenticated:
        st.divider()
        st.header("Account")
        st.success(f"Logged in: {st.session_state.user_id} ({st.session_state.role})")
        if st.button("Logout", use_container_width=True):
            do_logout()
            st.rerun()
    else:
        st.info("Login/Register is available on main page.")

    if st.session_state.is_authenticated and st.session_state.role == "user":
        st.divider()
        st.header("Response Controls")
        short_answer = st.toggle("Short Answer", value=False)
        detailed_explanation = st.toggle("Detailed Explanation", value=True)
        show_sources = st.toggle("Show Sources", value=True)

    if st.session_state.is_authenticated and st.session_state.role == "admin":
        st.divider()
        st.header("Admin Controls")
        admin_scope = st.selectbox("Analytics Scope", ["all", "current_user"], index=0)
        admin_user_filter = st.text_input("User ID Filter (optional)", value="")

if not backend_up:
    st.warning("FastAPI backend is not reachable. Start it from sidebar or run scripts/run_api.ps1.")
elif not st.session_state.is_authenticated:
    render_login_page(backend_up)
elif st.session_state.role == "user":
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask a B.Tech subject question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        mode = "short" if short_answer else "detailed"
        payload = {
            "message": prompt,
            "short_answer": short_answer,
            "detailed_explanation": detailed_explanation,
        }

        try:
            res = api_post("/chat", json_payload=payload, auth=True, timeout=420)
            if res.status_code == 401:
                do_logout()
                st.error("Session expired. Please login again.")
                st.stop()
            res.raise_for_status()
            data = res.json()
            answer = format_answer_blocks(data["response"])
            st.session_state.last_improved = bool(data.get("response_improved", False))
            st.session_state.last_variant = str(data.get("response_variant", ""))

            if show_sources and data.get("sources"):
                lines = [
                    f"- [{s.get('subject', 'unknown')}] {s['file_name']} (page {s['page_number']}, score={s['score']:.4f})"
                    for s in data["sources"]
                ]
                answer += "\n\nSources:\n" + "\n".join(lines)

            st.session_state.last_chat_id = data["chat_id"]
            st.session_state.last_mode = mode
        except requests.exceptions.ConnectionError:
            answer = "API error: backend is not running. Start backend from sidebar."
        except Exception as exc:
            answer = f"API error: {exc}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

    if st.session_state.last_improved:
        suffix = f" | Variant: {st.session_state.last_variant}" if st.session_state.last_variant else ""
        st.info(f"Response improvement indicator: adaptive low-reward correction is active for this query pattern.{suffix}")

    st.subheader("Rate Last Answer")
    cols = st.columns(5)
    for i in range(1, 6):
        if cols[i - 1].button(str(i), use_container_width=True):
            if st.session_state.last_chat_id is None:
                st.warning("Ask at least one question before rating.")
            else:
                payload = {
                    "chat_id": st.session_state.last_chat_id,
                    "rating": i,
                    "mode_used": st.session_state.last_mode,
                }
                try:
                    res = api_post("/feedback", json_payload=payload, auth=True, timeout=30)
                    if res.status_code == 401:
                        do_logout()
                        st.error("Session expired. Please login again.")
                        st.stop()
                    res.raise_for_status()
                    out = res.json()
                    st.success(
                        f"Saved. Reward: {out['reward']:.2f} | Preferred mode: {out['preference_profile']['preferred_mode']}"
                    )
                except Exception as exc:
                    st.error(f"Feedback failed: {exc}")

else:
    st.header("Admin Dashboard")

    docs = fetch_documents()
    users = fetch_users()

    scope_user = admin_user_filter.strip() if admin_scope == "current_user" and admin_user_filter.strip() else None
    analytics = fetch_analytics(scope=admin_scope, user_id=scope_user, weak_topics_limit=10000)
    reward_history = fetch_reward_history(user_id=scope_user if admin_scope == "current_user" else None, limit=3000)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", len(users))
    c2.metric("PDFs indexed", docs.get("total_pdfs", 0))
    c3.metric("Chunks indexed", docs.get("total_chunks", 0))
    c4.metric("Avg Reward", analytics.get("avg_reward", 0))

    if st.button("Rebuild Global Index", use_container_width=False):
        ok, msg = rebuild_index()
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(f"Rebuild failed: {msg}")

    st.subheader("User Database")
    if users:
        st.dataframe(pd.DataFrame(users), use_container_width=True)
    else:
        st.info("No users found.")

    st.subheader("Reward Analytics")
    if reward_history:
        df = pd.DataFrame(reward_history)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")

        df["rolling_reward"] = df["reward"].rolling(window=5, min_periods=1).mean()
        df["rolling_rating"] = df["rating"].rolling(window=5, min_periods=1).mean()
        df["cumulative_reward"] = df["reward"].cumsum()
        df["low_rating_flag"] = (df["rating"] <= 2).astype(int)
        df["low_rating_rate"] = df["low_rating_flag"].rolling(window=10, min_periods=1).mean() * 100.0

        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Total Chats", analytics.get("total_chats", 0))
        a2.metric("Total Feedback", analytics.get("total_feedback", 0))
        a3.metric("Avg Rating", f"{float(df['rating'].mean()):.2f}")
        a4.metric("Avg Reward", f"{float(df['reward'].mean()):.3f}")
        a5.metric("Low Rating %", f"{float((df['rating'] <= 2).mean() * 100):.1f}%")

        fig = px.line(df, x="created_at", y=["reward", "rolling_reward", "rolling_rating"], title="Reward & Rating Trend")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(df, x="created_at", y=["cumulative_reward", "low_rating_rate"], title="Cumulative Reward & Low-Rating Rate")
        st.plotly_chart(fig2, use_container_width=True)

        rating_counts = df["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["rating", "count"]
        fig3 = px.bar(rating_counts, x="rating", y="count", title="Rating Distribution")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Rated Questions Log**")
        log_cols = [c for c in ["created_at", "user_id", "query", "rating", "reward", "mode_used"] if c in df.columns]
        st.dataframe(df.sort_values("created_at", ascending=False)[log_cols], use_container_width=True)
    else:
        st.info("No reward history for selected scope.")

    weak_topics = analytics.get("weak_topics", [])
    st.subheader("Rewarded Query Patterns")
    if weak_topics:
        st.dataframe(pd.DataFrame(weak_topics), use_container_width=True)
    else:
        st.info("No query patterns found.")
