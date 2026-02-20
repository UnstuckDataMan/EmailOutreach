import re
import random
from datetime import date, datetime, time, timedelta
from io import BytesIO

import pandas as pd
import streamlit as st

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

try:
    # supabase-py v2
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None  # type: ignore


PLACEHOLDER_RE = re.compile(r"\{\{\s*([^\}]+?)\s*\}\}")


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
        # Also add underscore variant for mapping (e.g. company_name → Company Name)
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

    # de-dupe missing while preserving order
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
    # Remove consecutive commas (e.g., ,, -> ,)
    merged = re.sub(r',+', ',', merged)
    return merged


def find_email_column(df: pd.DataFrame) -> str | None:
    """Match any 'Email' variant (case/spaces/underscores)."""
    for col in df.columns:
        if norm_key(col) == "email":
            return col
    return None


def template_editor(title: str, session_key: str, min_templates: int = 1, help_text: str | None = None):
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
        if st.button("Remove last", key=f"rm_{session_key}") and len(st.session_state[session_key]) > min_templates:
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


def _format_time(dt: datetime) -> str:
    """Format like 1:02PM (no space)."""
    h = dt.hour % 12
    h = 12 if h == 0 else h
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{h}:{dt.minute:02d}{ampm}"


def build_time_schedule(
    n_rows: int,
    recipient_tz_name: str,
    sender_tz_name: str,
    start_t: time,
    end_t: time,
    min_step_min: int = 2,
    max_step_min: int = 5,
) -> list[str]:
    """Build n_rows send times.

    The time window (start_t/end_t) is interpreted in the RECIPIENT local time zone.
    The output strings are converted to the SENDER local time zone (what the SDR should do).

    Rolls to next day if needed.
    """
    tz_rec = ZoneInfo(recipient_tz_name) if ZoneInfo else None
    tz_send = ZoneInfo(sender_tz_name) if ZoneInfo else None

    min_step_min = max(1, int(min_step_min))
    max_step_min = max(min_step_min, int(max_step_min))

    out: list[str] = []
    day = date.today()

    while len(out) < n_rows:
        rec_start_dt = datetime.combine(day, start_t)
        rec_end_dt = datetime.combine(day, end_t)

        if tz_rec:
            rec_start_dt = rec_start_dt.replace(tzinfo=tz_rec)
            rec_end_dt = rec_end_dt.replace(tzinfo=tz_rec)

        cur = rec_start_dt
        while len(out) < n_rows and cur <= rec_end_dt:
            if tz_rec and tz_send:
                send_dt = cur.astimezone(tz_send)
                out.append(_format_time(send_dt))
            else:
                # Fallback (shouldn't happen on Streamlit Cloud): output recipient-local time
                out.append(_format_time(cur))

            cur += timedelta(minutes=random.randint(min_step_min, max_step_min))

        day += timedelta(days=1)

    return out

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


def build_sender_sequence(senders: list[str], n_rows: int, repeats_per_sender: int = 10) -> list[str]:
    if not senders:
        return [""] * n_rows
    repeats_per_sender = max(1, int(repeats_per_sender))

    # First, produce one block per sender (sender1 x N, sender2 x N, ...)
    seq: list[str] = []
    for s in senders:
        seq.extend([s] * repeats_per_sender)

    # If more rows are required, continue cycling to fill remaining rows
    if len(seq) < n_rows:
        i = 0
        while len(seq) < n_rows:
            sender = senders[i % len(senders)]
            seq.extend([sender] * repeats_per_sender)
            i += 1

    return seq[:n_rows]


def build_time_schedule_for_senders(
    senders: list[str],
    n_rows: int,
    recipient_tz_name: str,
    sender_tz_name: str,
    start_t: time,
    end_t: time,
    repeats_per_sender: int = 10,
    sender_start_interval_min: int = 30,
    min_step_min: int = 2,
    max_step_min: int = 5,
):
    """Build times aligned to sender blocks.

    Produces times in blocks: for each sender produce `repeats_per_sender` times
    starting at the recipient-local `start_t` plus a per-sender start interval
    (e.g., 30 minutes) to ensure ordering. Continues across multiple days as needed.
    """
    if not senders:
        return build_time_schedule(n_rows, recipient_tz_name, sender_tz_name, start_t, end_t, min_step_min, max_step_min)

    tz_rec = ZoneInfo(recipient_tz_name) if ZoneInfo else None
    tz_send = ZoneInfo(sender_tz_name) if ZoneInfo else None

    min_step_min = max(1, int(min_step_min))
    max_step_min = max(min_step_min, int(max_step_min))
    repeats_per_sender = max(1, int(repeats_per_sender))

    out: list[str] = []
    day = date.today()

    # Base recipient-local start datetime
    base_start_dt = datetime.combine(day, start_t)
    if tz_rec:
        base_start_dt = base_start_dt.replace(tzinfo=tz_rec)

    # Generate one block per sender on day 1
    rec_end_dt = datetime.combine(day, end_t)
    if tz_rec:
        rec_end_dt = rec_end_dt.replace(tzinfo=tz_rec)

    for si, _s in enumerate(senders):
        # per-sender offset (in minutes) - deterministic and monotonically increasing
        # Each sender starts `sender_start_interval_min` minutes after the previous
        offset = si * int(sender_start_interval_min)
        sender_start = base_start_dt + timedelta(minutes=offset)
        cur = sender_start
        while len(out) < n_rows and len(out) < (si + 1) * repeats_per_sender and cur <= rec_end_dt:
            if tz_rec and tz_send:
                send_dt = cur.astimezone(tz_send)
                out.append(_format_time(send_dt))
            else:
                out.append(_format_time(cur))
            cur += timedelta(minutes=random.randint(min_step_min, max_step_min))

    # If we've filled all rows, return
    if len(out) >= n_rows:
        return out[:n_rows]

    # Continue on subsequent days with fresh sender blocks
    current_day = day + timedelta(days=1)
    while len(out) < n_rows:
        rec_start_day = datetime.combine(current_day, start_t)
        rec_end_day = datetime.combine(current_day, end_t)
        if tz_rec:
            rec_start_day = rec_start_day.replace(tzinfo=tz_rec)
            rec_end_day = rec_end_day.replace(tzinfo=tz_rec)

        # Generate one block per sender on this day
        for si, _s in enumerate(senders):
            if len(out) >= n_rows:
                break
            # per-sender offset (in minutes) - deterministic and monotonically increasing
            offset = si * int(sender_start_interval_min)
            sender_start = rec_start_day + timedelta(minutes=offset)
            cur = sender_start
            while len(out) < n_rows and len(out) < ((current_day - day).days * len(senders) * repeats_per_sender) + ((si + 1) * repeats_per_sender):
                if cur > rec_end_day:
                    break
                if tz_rec and tz_send:
                    send_dt = cur.astimezone(tz_send)
                    out.append(_format_time(send_dt))
                else:
                    out.append(_format_time(cur))
                cur += timedelta(minutes=random.randint(min_step_min, max_step_min))

        current_day += timedelta(days=1)

    # If still short, pad with empty strings
    if len(out) < n_rows:
        out.extend([""] * (n_rows - len(out)))

    return out[:n_rows]


@st.cache_resource(show_spinner=False)
def get_supabase_client():
    """Returns a Supabase client if secrets are configured, else None."""
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


# ---------------- UI ----------------

st.set_page_config(page_title="Outreach Merge Tool", layout="centered")
st.title("Outreach Merge Tool")

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
blank_fill = st.text_input("Blank cell replacement", value="[MISSING]")
st.caption("If a cell is blank/empty, it becomes the value above (use empty string if you prefer).")

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
    help_text="Optional follow-up copy. If multiple are provided, rotates A → B → A…",
)

# LinkedIn templates
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

st.subheader("Schedule & Sender Settings")

# Choose the locale you are sending TO (recipient timezone) and the timezone you will work in.
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
    index=list(RECIPIENT_TZ_OPTIONS.keys()).index("UK") if "UK" in RECIPIENT_TZ_OPTIONS else 0,
)
recipient_tz_name = RECIPIENT_TZ_OPTIONS[recipient_label]

sender_label = st.radio(
    "Your timezone (Time column output)",
    options=["UK", "South Africa", "North Macedonia"],
    horizontal=True,
)
if sender_label == "UK":
    sender_tz_name = "Europe/London"
elif sender_label == "South Africa":
    sender_tz_name = "Africa/Johannesburg"
else:  # North Macedonia
    sender_tz_name = "Europe/Skopje"

time_cols = st.columns(2)
with time_cols[0]:
    start_time = st.time_input("Recipient-local start time", value=time(8, 30))
with time_cols[1]:
    end_time = st.time_input("Recipient-local end time", value=time(16, 0))

step_cols = st.columns(2)
with step_cols[0]:
    min_step = st.number_input("Min gap (minutes)", min_value=1, max_value=60, value=10, step=1)
with step_cols[1]:
    max_step = st.number_input("Max gap (minutes)", min_value=1, max_value=60, value=20, step=1)

repeats_per_sender = st.number_input(
    "Rows per sender before cycling (e.g., 10)",
    min_value=1,
    max_value=500,
    value=10,
    step=1,
)

sender_start_interval = st.number_input(
    "Sender start interval (minutes) (e.g. 30)",
    min_value=1,
    max_value=24 * 60,
    value=30,
    step=1,
)

sb = get_supabase_client()
sender_mode_options = ["Saved profile (Supabase)", "Upload / paste list"]
if sb is None:
    sender_mode_options = ["Upload / paste list"]
    st.info("Supabase not configured yet (add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Streamlit Secrets) — using upload/paste.")

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
            index=(profiles.index(st.session_state.selected_profile) if profiles and st.session_state.selected_profile in profiles else 0),
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
            if ok:
                st.success(msg)
            else:
                st.error(msg)
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
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.session_state.selected_profile = nm
                st.rerun()

    with crud[3]:
        if st.button("Delete selected", type="secondary") and selected:
            ok, msg = sb_delete_profile(sb, selected)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.session_state.selected_profile = ""
            st.rerun()

    senders_from_profile = parse_sender_gmails(emails_text)

else:
    # --- upload / paste fallback ---
    sender_file = st.file_uploader(
        "Upload sender gmails (txt/csv; one email per line)",
        type=["txt", "csv"],
    )
    sender_paste = st.text_area(
        "Or paste sender gmails (one per line)",
        height=120,
        placeholder="leo@domain.com\nanother@domain.com",
    )
    senders_from_profile = parse_sender_gmails(sender_file, sender_paste)
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

if run:
    if start_time >= end_time:
        st.error("Start time must be before end time.")
        st.stop()

    # Read Excel
    try:
        df = pd.read_excel(uploaded, dtype=object)
    except Exception as e:
        st.error(f"Could not read Excel: {e}")
        st.stop()

    header_map = build_header_map(df)

    # Validate placeholders across subject + email + chaser
    all_templates = subject_templates + email_templates + chaser_templates + linkedin_conn_templates + linkedin_msg_templates
    mapping, missing_placeholders = validate_mappings(all_templates, header_map)
    if missing_placeholders:
        st.error("Some placeholders do not match any Excel column header (case-insensitive; ignores spaces/underscores).")
        st.code("\n".join([f"UNMAPPED PLACEHOLDER: {{{{{ph}}}}}" for ph in missing_placeholders]))
        st.stop()

    email_col = find_email_column(df)

    # Sender list and Time + Sender columns
    sender_list = senders_from_profile if sender_mode.startswith("Saved") else senders_from_upload
    
    # Calculate where the break note should be inserted (after day 1 complete)
    repeats_val = int(repeats_per_sender)
    rows_per_day = len(sender_list) * repeats_val if sender_list else len(df)
    break_row_index = min(rows_per_day, len(df))  # Insert after day 1 or at end if df is smaller
    
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
    out_lead_2: list[str] = []

    # Build times for all rows (adjusted for break row)
    times_for_all = build_time_schedule_for_senders(
        senders=sender_list,
        n_rows=len(df) + 1,  # Request one extra to account for break row
        recipient_tz_name=recipient_tz_name,
        sender_tz_name=sender_tz_name,
        start_t=start_time,
        end_t=end_time,
        repeats_per_sender=repeats_val,
        sender_start_interval_min=int(sender_start_interval),
        min_step_min=int(min_step),
        max_step_min=int(max_step),
    )
    
    # Build senders for all rows
    senders_for_all = build_sender_sequence(sender_list, n_rows=len(df) + 1, repeats_per_sender=repeats_val)

    for i in range(len(df)):
        # Insert break row if we've reached the break point
        if i == break_row_index and len(df) > break_row_index:
            out_time.append("No more emails for today delay to next day")
            out_sender.append("")
            out_email_address.append("")
            out_subject.append("")
            out_email_copy.append("")
            out_email_sent.append("")
            out_chaser_copy.append("")
            out_chaser_sent.append("")
            out_lead.append("")
            out_user_linkedin.append("")
            out_linkedin_conn.append("")
            out_linkedin_msg.append("")
            out_lead_2.append("")
        
        # Get time and sender (adjust index if we're past break row)
        time_idx = i + 1 if i >= break_row_index else i
        out_time.append(times_for_all[time_idx] if time_idx < len(times_for_all) else "")
        out_sender.append(senders_for_all[time_idx] if time_idx < len(senders_for_all) else "")

        row = df.iloc[i]

        subj_t = subject_templates[i % len(subject_templates)]
        body_t = email_templates[i % len(email_templates)]
        chaser_t = chaser_templates[i % len(chaser_templates)] if len(chaser_templates) > 0 else ""
        linkedin_conn_t = linkedin_conn_templates[i % len(linkedin_conn_templates)] if len(linkedin_conn_templates) > 0 else ""
        linkedin_msg_t = linkedin_msg_templates[i % len(linkedin_msg_templates)] if len(linkedin_msg_templates) > 0 else ""

        subject_line = merge_row(subj_t, row, mapping, blank_fill)
        email_copy = merge_row(body_t, row, mapping, blank_fill)
        chaser_copy = merge_row(chaser_t, row, mapping, blank_fill) if chaser_t else ""
        linkedin_conn = merge_row(linkedin_conn_t, row, mapping, blank_fill) if linkedin_conn_t else ""
        linkedin_msg = merge_row(linkedin_msg_t, row, mapping, blank_fill) if linkedin_msg_t else ""

        if email_col:
            v = row.get(email_col, "")
            out_email_address.append("" if pd.isna(v) else str(v))
        else:
            out_email_address.append("")

        # User LinkedIn: find column matching 'Person Linkedin Url' (case/space/underscore-insensitive)
        user_linkedin = ""
        linkedin_col = None
        for col in df.columns:
            if norm_key(col) == "personlinkedinurl":
                linkedin_col = col
                break
        if linkedin_col:
            v = row.get(linkedin_col, "")
            user_linkedin = "" if pd.isna(v) else str(v)
        out_user_linkedin.append(user_linkedin)

        out_subject.append(subject_line)
        out_email_copy.append(email_copy)
        out_email_sent.append("")
        out_chaser_copy.append(chaser_copy)
        out_chaser_sent.append("")
        out_lead.append("")
        out_linkedin_conn.append(linkedin_conn)
        out_linkedin_msg.append(linkedin_msg)
        out_lead_2.append("")

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
            "Lead? 2": out_lead_2,
        }
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Outreach")
        ws = writer.sheets["Outreach"]

        # Define formats for conditional formatting
        yellow_format = writer.book.add_format({'bg_color': 'yellow'})
        light_blue_format = writer.book.add_format({'bg_color': '#ADD8E6'})  # Light blue
        green_format = writer.book.add_format({'bg_color': 'green'})
        pink_format = writer.book.add_format({'bg_color': '#FFC0CB'})  # Pink
        red_format = writer.book.add_format({'bg_color': 'red'})

        # Column widths
        for col_idx, col_name in enumerate(out_df.columns):
            sample = out_df[col_name].astype(str).head(50)
            max_len = max([len(col_name)] + [len(x) for x in sample])
            ws.set_column(col_idx, col_idx, min(max(12, max_len + 2), 60))

        # Specific widths
        ws.set_column(out_df.columns.get_loc("Time"), out_df.columns.get_loc("Time"), 10)
        ws.set_column(out_df.columns.get_loc("Sender Gmail"), out_df.columns.get_loc("Sender Gmail"), 28)

        email_sent_col = out_df.columns.get_loc("Email Sent?")
        chaser_sent_col = out_df.columns.get_loc("Chaser sent?")
        lead_col = out_df.columns.get_loc("Lead?")
        lead_col_2 = out_df.columns.get_loc("Lead? 2")

        first_row = 1
        last_row = len(out_df)

        # Yes/No dropdowns
        ws.data_validation(first_row, email_sent_col, last_row, email_sent_col, {"validate": "list", "source": ["No", "Yes"]})
        ws.data_validation(first_row, chaser_sent_col, last_row, chaser_sent_col, {"validate": "list", "source": ["No", "Yes"]})

        # Lead? dropdown
        ws.data_validation(first_row, lead_col, last_row, lead_col, {"validate": "list", "source": ["Lead", "Replied", "Unsubscribed"]})
        ws.data_validation(first_row, lead_col_2, last_row, lead_col_2, {"validate": "list", "source": ["Lead", "Replied", "Unsubscribed"]})

        # Default sent columns to No
        for r in range(first_row, last_row + 1):
            ws.write(r, email_sent_col, "No")
            ws.write(r, chaser_sent_col, "No")

        # Conditional formatting
        # Email Sent?: Yellow for "No", Blue for "Yes"
        ws.conditional_format(first_row, email_sent_col, last_row, email_sent_col, {
            'type': 'cell',
            'criteria': '==',
            'value': '"No"',
            'format': yellow_format
        })
        ws.conditional_format(first_row, email_sent_col, last_row, email_sent_col, {
            'type': 'cell',
            'criteria': '==',
            'value': '"Yes"',
            'format': light_blue_format
        })

        # Chaser sent?: Yellow for "No", Blue for "Yes"
        ws.conditional_format(first_row, chaser_sent_col, last_row, chaser_sent_col, {
            'type': 'cell',
            'criteria': '==',
            'value': '"No"',
            'format': yellow_format
        })
        ws.conditional_format(first_row, chaser_sent_col, last_row, chaser_sent_col, {
            'type': 'cell',
            'criteria': '==',
            'value': '"Yes"',
            'format': light_blue_format
        })

        # Lead?: Green for "Lead", Pink for "Replied", Red for "Unsubscribed"
        ws.conditional_format(first_row, lead_col, last_row, lead_col, {
            'type': 'cell',
            'criteria': '==',
            'value': '"Lead"',
            'format': green_format
        })
        ws.conditional_format(first_row, lead_col, last_row, lead_col, {
            'type': 'cell',
            'criteria': '==',
            'value': '"Replied"',
            'format': pink_format
        })
        ws.conditional_format(first_row, lead_col, last_row, lead_col, {
            'type': 'cell',
            'criteria': '==',
            'value': '"Unsubscribed"',
            'format': red_format
        })

        # Lead? 2: Same formatting as Lead?
        ws.conditional_format(first_row, lead_col_2, last_row, lead_col_2, {
            'type': 'cell',
            'criteria': '==',
            'value': '"Lead"',
            'format': green_format
        })
        ws.conditional_format(first_row, lead_col_2, last_row, lead_col_2, {
            'type': 'cell',
            'criteria': '==',
            'value': '"Replied"',
            'format': pink_format
        })
        ws.conditional_format(first_row, lead_col_2, last_row, lead_col_2, {
            'type': 'cell',
            'criteria': '==',
            'value': '"Unsubscribed"',
            'format': red_format
        })

        # Highlight the first row for each sender change (when Sender Gmail is non-empty
        # and different from the previous row). This makes it easier to spot sender blocks.
        sender_col = out_df.columns.get_loc("Sender Gmail")

        def _col_idx_to_letter(idx: int) -> str:
            """Convert zero-based column index to Excel column letters (0 -> A)."""
            letters = ""
            while idx >= 0:
                letters = chr(ord('A') + (idx % 26)) + letters
                idx = idx // 26 - 1
            return letters

        sender_col_letter = _col_idx_to_letter(sender_col)

        # Formula applied to the full data row range. Using $<col><row> references so the row
        # portion adjusts for each row in the applied range. Top data row is Excel row 2.
        formula = f"=AND(${sender_col_letter}2<>\"\", ${sender_col_letter}2<>${sender_col_letter}1)"

        # Apply the sender-change highlight to the Sender Gmail column only
        first_col = 0
        last_col = len(out_df.columns) - 1
        ws.conditional_format(first_row, sender_col, last_row, sender_col, {
            'type': 'formula',
            'criteria': formula,
            'format': yellow_format,
        })

    buffer.seek(0)
    st.success(f"Done. Generated {len(out_df)} rows.")
    st.download_button(
        label="Download List",
        data=buffer.getvalue(),
        file_name=output_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
