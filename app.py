import re
from datetime import date, time
from io import BytesIO

import pandas as pd
import streamlit as st

try:
    from supabase import create_client
except Exception:
    create_client = None  # type: ignore

from scheduler import (
    ScheduledSend,
    SchedulerError,
    build_month_schedule,
    compute_daily_targets,
    compute_max_senders_per_day,
    fetch_bank_holidays,
    get_working_days,
)

PLACEHOLDER_RE = re.compile(r"\{\{\s*([^\}]+?)\s*\}\}")


# ---------------------------------------------------------------------------
# Template merge helpers
# ---------------------------------------------------------------------------

def norm_key(s: str) -> str:
    """Case-insensitive, ignore spaces + underscores."""
    return re.sub(r"[ _]+", "", str(s).strip().lower())


def extract_placeholders(template: str) -> list[str]:
    return PLACEHOLDER_RE.findall(template)


def build_header_map(df: pd.DataFrame) -> dict[str, str]:
    """{normalized_header: original_header} (first wins)."""
    m: dict[str, str] = {}
    for col in df.columns:
        k = norm_key(col)
        if k not in m:
            m[k] = col
        k_us = k.replace(" ", "_")
        if k_us not in m:
            m[k_us] = col
    return m


def validate_mappings(all_templates: list[str], header_map: dict[str, str]):
    """Validate placeholders across all templates."""
    all_placeholders: list[str] = []
    for t in all_templates:
        all_placeholders.extend(extract_placeholders(t))

    missing: list[str] = []
    mapping: dict[str, str] = {}
    for ph in all_placeholders:
        key = norm_key(ph)
        if key in header_map:
            mapping[ph] = header_map[key]
        else:
            missing.append(ph)

    seen: set[str] = set()
    missing_unique: list[str] = []
    for m in missing:
        if m not in seen:
            missing_unique.append(m)
            seen.add(m)
    return mapping, missing_unique


def merge_row(template: str, row: pd.Series, mapping: dict[str, str], blank_fill: str) -> str:
    def repl(match: re.Match) -> str:
        raw = match.group(1)
        col = mapping.get(raw)
        if not col:
            return ""
        val = row.get(col, "")
        if pd.isna(val) or str(val).strip() == "":
            return blank_fill
        return str(val)

    merged = PLACEHOLDER_RE.sub(repl, template)
    merged = re.sub(r",+", ",", merged)
    return merged


def find_email_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if norm_key(col) == "email":
            return col
    return None


# ---------------------------------------------------------------------------
# Template editor widget
# ---------------------------------------------------------------------------

def template_editor(
    title: str,
    session_key: str,
    min_templates: int = 1,
    help_text: str | None = None,
):
    """Dynamic list of templates stored in st.session_state[session_key]."""
    st.subheader(title)
    if help_text:
        st.caption(help_text)

    if session_key not in st.session_state:
        st.session_state[session_key] = [""] * max(1, min_templates)

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button(f"Add {title.lower()}", key=f"add_{session_key}"):
            st.session_state[session_key].append("")
    with cols[1]:
        if (
            st.button("Remove last", key=f"rm_{session_key}")
            and len(st.session_state[session_key]) > min_templates
        ):
            st.session_state[session_key].pop()

    for i in range(len(st.session_state[session_key])):
        label = f"{title} {chr(65 + i)}"
        st.session_state[session_key][i] = st.text_area(
            label,
            value=st.session_state[session_key][i],
            height=120,
            placeholder="Use {{placeholders}} like {{first_name}}",
            key=f"{session_key}_{i}",
        )

    return [t.strip() for t in st.session_state[session_key] if t.strip()]


# ---------------------------------------------------------------------------
# Sender helpers
# ---------------------------------------------------------------------------

def parse_sender_gmails(raw: str) -> list[str]:
    raw = raw or ""
    lines = [ln.strip() for ln in raw.replace(",", "\n").splitlines()]
    lines = [ln for ln in lines if ln]
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        if ln not in seen:
            out.append(ln)
            seen.add(ln)
    return out


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_supabase_client():
    if create_client is None:
        return None
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def sb_list_profiles(sb) -> list[str]:
    try:
        res = sb.table("sender_profiles").select("profile_name").order("profile_name").execute()
        return [r["profile_name"] for r in (res.data or [])]
    except Exception:
        return []


def sb_get_profile_emails(sb, profile_name: str) -> list[str]:
    try:
        res = (
            sb.table("sender_profiles")
            .select("emails")
            .eq("profile_name", profile_name)
            .limit(1)
            .execute()
        )
        if not res.data:
            return []
        emails = res.data[0].get("emails", [])
        if isinstance(emails, list):
            return [str(x) for x in emails if str(x).strip()]
        return []
    except Exception:
        return []


def sb_upsert_profile(sb, profile_name: str, emails: list[str]) -> tuple[bool, str]:
    try:
        sb.table("sender_profiles").upsert({"profile_name": profile_name, "emails": emails}).execute()
        return True, "Saved."
    except Exception as e:
        return False, f"Save failed: {e}"


def sb_delete_profile(sb, profile_name: str) -> tuple[bool, str]:
    try:
        sb.table("sender_profiles").delete().eq("profile_name", profile_name).execute()
        return True, "Deleted."
    except Exception as e:
        return False, f"Delete failed: {e}"


# ---------------------------------------------------------------------------
# Bank holidays (cached, refreshes once per hour)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_bank_holidays() -> tuple[set[date], str | None]:
    try:
        return fetch_bank_holidays(), None
    except SchedulerError as exc:
        return set(), str(exc)


# ---------------------------------------------------------------------------
# Excel column letter helper
# ---------------------------------------------------------------------------

def _col_letter(idx: int) -> str:
    """Convert zero-based column index to Excel letter(s): 0→A, 25→Z, 26→AA."""
    letters = ""
    while idx >= 0:
        letters = chr(ord("A") + (idx % 26)) + letters
        idx = idx // 26 - 1
    return letters


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Outreach Merge Tool", layout="centered")
st.title("Outreach Merge Tool")

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
blank_fill = st.text_input("Blank cell replacement", value="[MISSING]")
st.caption("If a cell is blank/empty, it is replaced with the value above (use an empty string if you prefer).")

subject_templates = template_editor(
    "Subject Template",
    session_key="subject_templates",
    min_templates=1,
    help_text="One or more subject lines. Rotates A → B → A… across rows.",
)

email_templates = template_editor(
    "Email Copy Template",
    session_key="email_templates",
    min_templates=1,
    help_text="One or more main email bodies. Rotates A → B → A… across rows.",
)

chaser_templates = template_editor(
    "Chaser Copy Template",
    session_key="chaser_templates",
    min_templates=0,
    help_text="Optional follow-up copy. Rotates A → B → A… No send times are assigned to chasers.",
)

linkedin_conn_templates = template_editor(
    "LinkedIn Connection Template",
    session_key="linkedin_conn_templates",
    min_templates=0,
    help_text="Optional LinkedIn connection request copy. Rotates A → B → A… across rows.",
)
linkedin_msg_templates = template_editor(
    "LinkedIn Messaging Template",
    session_key="linkedin_msg_templates",
    min_templates=0,
    help_text="Optional LinkedIn message copy. Rotates A → B → A… across rows.",
)

# ---------------------------------------------------------------------------
# Schedule settings
# ---------------------------------------------------------------------------

st.subheader("Schedule Settings")

today = date.today()

sched_cols = st.columns(2)
with sched_cols[0]:
    month = st.selectbox(
        "Month",
        options=list(range(1, 13)),
        format_func=lambda m: date(2000, m, 1).strftime("%B"),
        index=today.month - 1,
    )
with sched_cols[1]:
    year = st.selectbox(
        "Year",
        options=[today.year, today.year + 1],
        index=0,
    )

RECIPIENT_TZ_OPTIONS = {
    "US Eastern (ET)": "America/New_York",
    "US Central (CT)": "America/Chicago",
    "US Mountain (MT)": "America/Denver",
    "US Pacific (PT)": "America/Los_Angeles",
    "UK": "Europe/London",
    "South Africa": "Africa/Johannesburg",
}

recipient_label = st.selectbox(
    "Recipient locale (send-to timezone)",
    options=list(RECIPIENT_TZ_OPTIONS.keys()),
    index=list(RECIPIENT_TZ_OPTIONS.keys()).index("UK"),
)
recipient_tz = RECIPIENT_TZ_OPTIONS[recipient_label]

sender_label = st.radio(
    "Your timezone (Time column output)",
    options=["UK", "South Africa", "North Macedonia"],
    horizontal=True,
)
if sender_label == "UK":
    sender_tz = "Europe/London"
elif sender_label == "South Africa":
    sender_tz = "Africa/Johannesburg"
else:
    sender_tz = "Europe/Skopje"

st.markdown("**Recipient-local day window**")
win_cols = st.columns(2)
with win_cols[0]:
    window_start = st.time_input("Window start", value=time(8, 30))
with win_cols[1]:
    window_end = st.time_input("Window end", value=time(15, 30))

sender1_start = st.time_input(
    "Sender 1 recipient-local start time",
    value=time(9, 5),
    help="Must be within the window above. Each subsequent sender starts 1 hour later.",
)

cap_cols = st.columns(2)
with cap_cols[0]:
    per_sender_cap = st.number_input(
        "Per-sender daily cap",
        min_value=1,
        max_value=500,
        value=15,
        step=1,
    )
with cap_cols[1]:
    st.markdown("&nbsp;", unsafe_allow_html=True)  # spacer

guard_cols = st.columns(2)
with guard_cols[0]:
    monthly_min = st.number_input("Monthly min prospects (warn only)", min_value=0, value=100, step=10)
with guard_cols[1]:
    monthly_max = st.number_input("Monthly max prospects (warn only)", min_value=0, value=1650, step=50)

# ---------------------------------------------------------------------------
# Sender source
# ---------------------------------------------------------------------------

sb = get_supabase_client()
sender_mode_options = ["Saved profile (Supabase)", "Upload / paste list"]
if sb is None:
    sender_mode_options = ["Upload / paste list"]
    st.info(
        "Supabase not configured (add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in "
        "Streamlit Secrets) — using upload/paste mode."
    )

sender_mode = st.radio("Sender Gmail source", options=sender_mode_options, horizontal=True)

senders_from_profile: list[str] = []
senders_from_upload: list[str] = []

if sender_mode.startswith("Saved") and sb is not None:
    st.markdown("#### Sender profiles")
    profiles = sb_list_profiles(sb)
    if "selected_profile" not in st.session_state:
        st.session_state.selected_profile = profiles[0] if profiles else ""

    top = st.columns([2, 1])
    with top[0]:
        selected = st.selectbox(
            "Choose profile",
            options=profiles if profiles else [""],
            index=(
                profiles.index(st.session_state.selected_profile)
                if profiles and st.session_state.selected_profile in profiles
                else 0
            ),
            key="profile_select",
        )
        st.session_state.selected_profile = selected
    with top[1]:
        if st.button("Refresh list"):
            st.rerun()

    current_emails = sb_get_profile_emails(sb, selected) if selected else []
    st.caption("Edit the list below (one email per line) then Save.")
    emails_text = st.text_area(
        "Sender emails",
        value="\n".join(current_emails),
        height=160,
        placeholder="leo@domain.com\nanother@domain.com",
        key="emails_editor",
    )

    crud = st.columns([1, 1, 1, 2])
    with crud[0]:
        if st.button("Save changes"):
            cleaned = parse_sender_gmails(emails_text)
            ok, msg = sb_upsert_profile(sb, selected.strip(), cleaned)
            st.success(msg) if ok else st.error(msg)
            st.rerun()
    with crud[1]:
        new_name = st.text_input("New profile name", value="", key="new_profile_name")
    with crud[2]:
        if st.button("Create new"):
            nm = new_name.strip()
            if not nm:
                st.error("Enter a profile name.")
            else:
                ok, msg = sb_upsert_profile(sb, nm, parse_sender_gmails(emails_text))
                st.success(msg) if ok else st.error(msg)
                st.session_state.selected_profile = nm
                st.rerun()
    with crud[3]:
        if st.button("Delete selected", type="secondary") and selected:
            ok, msg = sb_delete_profile(sb, selected)
            st.success(msg) if ok else st.error(msg)
            st.session_state.selected_profile = ""
            st.rerun()

    senders_from_profile = parse_sender_gmails(emails_text)

else:
    sender_file = st.file_uploader(
        "Upload sender gmails (txt/csv; one email per line)",
        type=["txt", "csv"],
    )
    sender_paste = st.text_area(
        "Or paste sender gmails (one per line)",
        height=120,
        placeholder="leo@domain.com\nanother@domain.com",
    )
    raw = ""
    if sender_file is not None:
        try:
            raw = sender_file.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            raw = ""
    if not raw:
        raw = sender_paste or ""
    senders_from_upload = parse_sender_gmails(raw)

output_name = st.text_input(
    "Output file name",
    value="outreach_output.xlsx",
    help="Must end with .xlsx",
).strip()
if not output_name.lower().endswith(".xlsx"):
    output_name += ".xlsx"

run = st.button(
    "Generate output XLSX",
    type="primary",
    disabled=(uploaded is None or len(subject_templates) == 0 or len(email_templates) == 0),
)

# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

if run:
    # --- Input validation ---
    if window_start >= window_end:
        st.error("Window start must be before window end.")
        st.stop()
    if sender1_start < window_start or sender1_start > window_end:
        st.error("Sender 1 start time must be within the day window.")
        st.stop()

    # --- Read Excel ---
    try:
        df = pd.read_excel(uploaded, dtype=object)
    except Exception as e:
        st.error(f"Could not read Excel: {e}")
        st.stop()

    n_prospects = len(df)

    # --- Monthly guardrail warnings ---
    if n_prospects < monthly_min:
        st.warning(f"Prospect count ({n_prospects}) is below the monthly minimum ({monthly_min}).")
    if n_prospects > monthly_max:
        st.warning(f"Prospect count ({n_prospects}) exceeds the monthly maximum ({monthly_max}).")

    # --- Placeholder validation ---
    header_map = build_header_map(df)
    all_templates = (
        subject_templates
        + email_templates
        + chaser_templates
        + linkedin_conn_templates
        + linkedin_msg_templates
    )
    mapping, missing_placeholders = validate_mappings(all_templates, header_map)
    if missing_placeholders:
        st.error(
            "Some placeholders do not match any Excel column header "
            "(case-insensitive; spaces/underscores ignored)."
        )
        st.code("\n".join([f"UNMAPPED PLACEHOLDER: {{{{{ph}}}}}" for ph in missing_placeholders]))
        st.stop()

    # --- Sender list ---
    sender_list = senders_from_profile if sender_mode.startswith("Saved") else senders_from_upload
    if not sender_list:
        st.warning("No senders configured – Time and Sender Gmail columns will be empty.")

    # --- Bank holidays ---
    bank_holidays, bh_err = load_bank_holidays()
    if bh_err:
        st.warning(f"Could not fetch bank holidays ({bh_err}). Proceeding without them.")

    # --- Preview: working days + daily distribution ---
    working_days = get_working_days(year, month, bank_holidays)
    if not working_days:
        st.error(f"No working days found in {date(year, month, 1).strftime('%B %Y')}.")
        st.stop()

    try:
        daily_targets = compute_daily_targets(n_prospects, working_days)
    except SchedulerError as exc:
        st.error(str(exc))
        st.stop()

    n_wd = len(working_days)
    max_s = compute_max_senders_per_day(window_end, sender1_start, len(sender_list)) if sender_list else 0

    with st.expander("Schedule preview", expanded=False):
        st.markdown(
            f"**{date(year, month, 1).strftime('%B %Y')}** — "
            f"{n_wd} working days, {n_prospects} prospects, "
            f"≈{n_prospects / n_wd:.1f} per day"
        )
        if sender_list:
            st.markdown(
                f"Senders available: {len(sender_list)}, "
                f"senders that fit in window: **{max_s}**, "
                f"per-sender daily cap: {per_sender_cap}"
            )
        preview_rows = [
            {"Date": d.strftime("%a %d %b"), "Target": t}
            for d, t in zip(working_days, daily_targets)
        ]
        st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

    # --- Build schedule ---
    schedule: list[ScheduledSend] = []
    if sender_list:
        try:
            schedule = build_month_schedule(
                n_prospects=n_prospects,
                senders=sender_list,
                year=year,
                month=month,
                bank_holidays=bank_holidays,
                window_start=window_start,
                window_end=window_end,
                sender1_start=sender1_start,
                per_sender_cap=per_sender_cap,
                recipient_tz=recipient_tz,
                sender_tz=sender_tz,
            )
        except SchedulerError as exc:
            st.error(f"Scheduling error: {exc}")
            st.stop()
    else:
        # No senders: produce empty schedule slots
        schedule = [ScheduledSend(send_date=date.today(), display_time="", sender="") for _ in range(n_prospects)]

    # --- Find special columns ---
    email_col = find_email_column(df)
    linkedin_col = next(
        (col for col in df.columns if norm_key(col) == "personlinkedinurl"),
        None,
    )

    # --- Build output rows ---
    out_time: list[str] = []
    out_sender: list[str] = []
    out_email_address: list[str] = []
    out_subject: list[str] = []
    out_email_copy: list[str] = []
    out_email_sent: list[str] = []
    out_chaser_copy: list[str] = []
    out_chaser_sent: list[str] = []
    out_lead: list[str] = []
    out_user_linkedin: list[str] = []
    out_linkedin_conn: list[str] = []
    out_linkedin_msg: list[str] = []
    out_linkedin_lead: list[str] = []

    for i, row in df.iterrows():
        idx = int(i)  # type: ignore[arg-type]
        send = schedule[idx]

        out_time.append(send.display_time)
        out_sender.append(send.sender)

        # Email address
        if email_col:
            v = row.get(email_col, "")
            out_email_address.append("" if pd.isna(v) else str(v))
        else:
            out_email_address.append("")

        # Templates (rotate round-robin)
        subj_t = subject_templates[idx % len(subject_templates)]
        body_t = email_templates[idx % len(email_templates)]
        chaser_t = chaser_templates[idx % len(chaser_templates)] if chaser_templates else ""
        conn_t = linkedin_conn_templates[idx % len(linkedin_conn_templates)] if linkedin_conn_templates else ""
        msg_t = linkedin_msg_templates[idx % len(linkedin_msg_templates)] if linkedin_msg_templates else ""

        out_subject.append(merge_row(subj_t, row, mapping, blank_fill))
        out_email_copy.append(merge_row(body_t, row, mapping, blank_fill))
        out_email_sent.append("")
        out_chaser_copy.append(merge_row(chaser_t, row, mapping, blank_fill) if chaser_t else "")
        out_chaser_sent.append("")
        out_lead.append("")

        # Person LinkedIn URL
        if linkedin_col:
            v = row.get(linkedin_col, "")
            out_user_linkedin.append("" if pd.isna(v) else str(v))
        else:
            out_user_linkedin.append("")

        out_linkedin_conn.append(merge_row(conn_t, row, mapping, blank_fill) if conn_t else "")
        out_linkedin_msg.append(merge_row(msg_t, row, mapping, blank_fill) if msg_t else "")
        out_linkedin_lead.append("")

    out_df = pd.DataFrame(
        {
            "Time": out_time,
            "Sender Gmail": out_sender,
            "Email address": out_email_address,
            "Subject line": out_subject,
            "Email Copy": out_email_copy,
            "Email Sent?": out_email_sent,
            "Chaser copy": out_chaser_copy,
            "Chaser sent?": out_chaser_sent,
            "Lead?": out_lead,
            "User LinkedIn": out_user_linkedin,
            "LinkedIn Connection": out_linkedin_conn,
            "LinkedIn Messaging": out_linkedin_msg,
            "LinkedIn Lead?": out_linkedin_lead,
        }
    )

    # --- Excel formatting ---
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Outreach")
        wb = writer.book
        ws = writer.sheets["Outreach"]

        yellow_fmt = wb.add_format({"bg_color": "yellow"})
        blue_fmt = wb.add_format({"bg_color": "#ADD8E6"})
        green_fmt = wb.add_format({"bg_color": "green"})
        pink_fmt = wb.add_format({"bg_color": "#FFC0CB"})
        red_fmt = wb.add_format({"bg_color": "red"})
        bold_fmt = wb.add_format({"bold": True})

        # Column widths (auto-fit from sample, bounded)
        for ci, col_name in enumerate(out_df.columns):
            sample = out_df[col_name].astype(str).head(50)
            max_len = max([len(col_name)] + [len(x) for x in sample])
            ws.set_column(ci, ci, min(max(12, max_len + 2), 60))
        # Override specific columns
        ws.set_column(out_df.columns.get_loc("Time"), out_df.columns.get_loc("Time"), 18)
        ws.set_column(out_df.columns.get_loc("Sender Gmail"), out_df.columns.get_loc("Sender Gmail"), 28)

        first_row = 1
        last_row = len(out_df)

        email_sent_col = out_df.columns.get_loc("Email Sent?")
        chaser_sent_col = out_df.columns.get_loc("Chaser sent?")
        lead_col = out_df.columns.get_loc("Lead?")
        linkedin_lead_col = out_df.columns.get_loc("LinkedIn Lead?")

        # Dropdowns
        ws.data_validation(first_row, email_sent_col, last_row, email_sent_col,
                           {"validate": "list", "source": ["No", "Yes"]})
        ws.data_validation(first_row, chaser_sent_col, last_row, chaser_sent_col,
                           {"validate": "list", "source": ["No", "Yes"]})
        ws.data_validation(first_row, lead_col, last_row, lead_col,
                           {"validate": "list", "source": ["Lead", "Replied", "Unsubscribed"]})
        ws.data_validation(first_row, linkedin_lead_col, last_row, linkedin_lead_col,
                           {"validate": "list", "source": ["Lead", "Replied", "Unsubscribed"]})

        # Default sent columns to "No"
        for r in range(first_row, last_row + 1):
            ws.write(r, email_sent_col, "No")
            ws.write(r, chaser_sent_col, "No")

        # Conditional formatting — Email Sent?
        ws.conditional_format(first_row, email_sent_col, last_row, email_sent_col,
                              {"type": "cell", "criteria": "==", "value": '"No"', "format": yellow_fmt})
        ws.conditional_format(first_row, email_sent_col, last_row, email_sent_col,
                              {"type": "cell", "criteria": "==", "value": '"Yes"', "format": blue_fmt})

        # Conditional formatting — Chaser sent?
        ws.conditional_format(first_row, chaser_sent_col, last_row, chaser_sent_col,
                              {"type": "cell", "criteria": "==", "value": '"No"', "format": yellow_fmt})
        ws.conditional_format(first_row, chaser_sent_col, last_row, chaser_sent_col,
                              {"type": "cell", "criteria": "==", "value": '"Yes"', "format": blue_fmt})

        # Conditional formatting — Lead?
        for col_idx in (lead_col, linkedin_lead_col):
            ws.conditional_format(first_row, col_idx, last_row, col_idx,
                                  {"type": "cell", "criteria": "==", "value": '"Lead"', "format": green_fmt})
            ws.conditional_format(first_row, col_idx, last_row, col_idx,
                                  {"type": "cell", "criteria": "==", "value": '"Replied"', "format": pink_fmt})
            ws.conditional_format(first_row, col_idx, last_row, col_idx,
                                  {"type": "cell", "criteria": "==", "value": '"Unsubscribed"', "format": red_fmt})

        # Sender-change highlight: yellow when Sender Gmail changes from the row above.
        # Uses relative row references (no $ on row numbers) so Excel adjusts per row.
        sender_col_idx = out_df.columns.get_loc("Sender Gmail")
        scl = _col_letter(sender_col_idx)
        # Formula anchored to first data row (Excel row 2); row refs are relative so
        # they shift for each subsequent row in the conditional format range.
        sender_formula = f'=AND({scl}2<>"",{scl}2<>{scl}1)'
        ws.conditional_format(first_row, sender_col_idx, last_row, sender_col_idx,
                              {"type": "formula", "criteria": sender_formula, "format": yellow_fmt})

    buffer.seek(0)
    st.success(f"Done. Generated {len(out_df)} rows across {len(working_days)} working days.")
    st.download_button(
        label="Download List",
        data=buffer.getvalue(),
        file_name=output_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
