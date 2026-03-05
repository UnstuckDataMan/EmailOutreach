"""
Month-based email outreach scheduler.

Pure functions – no Streamlit or UI imports. Safe to import and test independently.
"""
from __future__ import annotations

import json
import random
from datetime import date, datetime, time, timedelta
from typing import NamedTuple
from urllib.error import URLError
from urllib.request import urlopen

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore

BANK_HOLIDAYS_URL = "https://www.gov.uk/bank-holidays.json"


class SchedulerError(Exception):
    pass


class ScheduledSend(NamedTuple):
    send_date: date
    display_time: str   # e.g. "Mon 03/03 9:05AM"  (sender-local)
    sender: str


# ---------------------------------------------------------------------------
# Bank holidays
# ---------------------------------------------------------------------------

def fetch_bank_holidays(division: str = "england-and-wales") -> set[date]:
    """
    Fetch England & Wales bank holiday dates from the GOV.UK JSON endpoint.
    Returns a set of date objects.
    Raises SchedulerError on network or parse failures.
    """
    try:
        with urlopen(BANK_HOLIDAYS_URL, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        raise SchedulerError(f"Network error fetching bank holidays: {exc}") from exc
    except Exception as exc:
        raise SchedulerError(f"Error parsing bank holidays response: {exc}") from exc

    events = data.get(division, {}).get("events", [])
    return {date.fromisoformat(e["date"]) for e in events}


# ---------------------------------------------------------------------------
# Working day computation
# ---------------------------------------------------------------------------

def get_working_days(year: int, month: int, bank_holidays: set[date]) -> list[date]:
    """
    Return a sorted list of Mon–Fri non-bank-holiday days in the given month/year.
    """
    result: list[date] = []
    d = date(year, month, 1)
    while d.month == month:
        if d.weekday() < 5 and d not in bank_holidays:  # Mon=0 … Fri=4
            result.append(d)
        d += timedelta(days=1)
    return result


# ---------------------------------------------------------------------------
# Prospect distribution
# ---------------------------------------------------------------------------

def compute_daily_targets(n_prospects: int, working_days: list[date]) -> list[int]:
    """
    Spread n_prospects exactly across working_days.
    The first (n % D) days receive (n // D + 1) prospects; the rest receive (n // D).
    """
    D = len(working_days)
    if D == 0:
        raise SchedulerError("No working days found in the selected month.")
    if n_prospects == 0:
        return [0] * D
    base, rem = divmod(n_prospects, D)
    return [base + (1 if i < rem else 0) for i in range(D)]


# ---------------------------------------------------------------------------
# Sender capacity
# ---------------------------------------------------------------------------

def compute_max_senders_per_day(
    window_end: time,
    sender1_start: time,
    n_senders: int,
) -> int:
    """
    Count how many senders fit in the window with 1-hour start offsets.
    Sender k's start = sender1_start + k*60 min; valid only if <= window_end.
    """
    base_dt = datetime.combine(date.today(), sender1_start)
    end_dt = datetime.combine(date.today(), window_end)
    count = 0
    for k in range(n_senders):
        if base_dt + timedelta(minutes=k * 60) <= end_dt:
            count += 1
        else:
            break
    return count


def allocate_senders_for_day(
    day_target: int,
    n_available: int,
    per_sender_cap: int,
) -> list[int]:
    """
    Distribute day_target sends across n_available senders, each capped at per_sender_cap.
    Allocations differ by at most 1 where feasible.
    Raises SchedulerError if total capacity < day_target.
    """
    capacity = n_available * per_sender_cap
    if day_target > capacity:
        raise SchedulerError(
            f"Insufficient daily capacity: target {day_target} exceeds "
            f"{n_available} senders × {per_sender_cap} cap = {capacity}."
        )
    base, rem = divmod(day_target, n_available)
    allocs = [min(base + (1 if i < rem else 0), per_sender_cap) for i in range(n_available)]

    # Redistribute any shortfall caused by per_sender_cap clipping
    shortfall = day_target - sum(allocs)
    for i in range(n_available):
        if shortfall <= 0:
            break
        room = per_sender_cap - allocs[i]
        take = min(room, shortfall)
        allocs[i] += take
        shortfall -= take

    if sum(allocs) != day_target:
        raise SchedulerError(
            f"Could not allocate {day_target} sends across {n_available} "
            f"senders with cap {per_sender_cap}."
        )
    return allocs


# ---------------------------------------------------------------------------
# Baseline slot generation
# ---------------------------------------------------------------------------

def _dt(t: time) -> datetime:
    """Combine today's date with a time for arithmetic convenience."""
    return datetime.combine(date.today(), t)


def generate_baseline_slots(
    sender_start: time,
    window_end: time,
    n_slots: int,
) -> list[time]:
    """
    Generate n_slots baseline times spaced 30 minutes apart: start, start+30m, …
    All slots must be <= window_end.
    Raises SchedulerError if the window is too narrow.
    """
    end_dt = _dt(window_end)
    slots: list[time] = []
    for i in range(n_slots):
        candidate = _dt(sender_start) + timedelta(minutes=30 * i)
        if candidate > end_dt:
            raise SchedulerError(
                f"Cannot fit {n_slots} baseline slots (30-min apart) for a sender "
                f"starting at {sender_start} before window end {window_end}. "
                "Increase the window, reduce the per-sender daily cap, or add more senders."
            )
        slots.append(candidate.time())
    return slots


# ---------------------------------------------------------------------------
# Variance + no-overlap
# ---------------------------------------------------------------------------

def apply_variance_no_overlap(
    baseline_slots: list[time],
    occupied: set[int],          # minute-of-day values (h*60+m) already reserved this day
    window_end: time,
    variance_min: int = 4,
    variance_max: int = 8,
    max_retries: int = 10,
) -> tuple[list[time], set[int]]:
    """
    Jitter each baseline slot by +variance_min..+variance_max minutes.
    Guarantees no two resulting times share the same minute-of-day.

    Falls back to forward-shifting if random retries are exhausted.
    Returns (jittered_times, updated_occupied).
    Raises SchedulerError if the window has no free minute for a slot.
    """
    result: list[time] = []
    end_dt = _dt(window_end)

    for slot in baseline_slots:
        base_dt = _dt(slot)
        placed = False

        # Attempt random variance up to max_retries times
        for _ in range(max_retries):
            var = random.randint(variance_min, variance_max)
            candidate = base_dt + timedelta(minutes=var)
            if candidate > end_dt:
                continue
            key = candidate.hour * 60 + candidate.minute
            if key not in occupied:
                occupied.add(key)
                result.append(candidate.time())
                placed = True
                break

        # Forward-shift fallback: walk minute-by-minute from base+variance_min
        if not placed:
            candidate = base_dt + timedelta(minutes=variance_min)
            while candidate <= end_dt:
                key = candidate.hour * 60 + candidate.minute
                if key not in occupied:
                    occupied.add(key)
                    result.append(candidate.time())
                    placed = True
                    break
                candidate += timedelta(minutes=1)

        if not placed:
            raise SchedulerError(
                f"No free minute available in the window after slot {slot}. "
                "Window is fully saturated – widen the window or reduce daily targets."
            )

    return result, occupied


# ---------------------------------------------------------------------------
# Time formatting / timezone conversion
# ---------------------------------------------------------------------------

def _fmt(t: time) -> str:
    h = t.hour % 12 or 12
    return f"{h}:{t.minute:02d}{'AM' if t.hour < 12 else 'PM'}"


def to_sender_time_str(
    d: date,
    t: time,
    recipient_tz: str,
    sender_tz: str,
) -> str:
    """
    Convert a recipient-local (date, time) pair to a formatted sender-local time string.
    Falls back to the recipient-local time if zoneinfo is unavailable.
    """
    if ZoneInfo is None:
        return _fmt(t)
    try:
        rec_dt = datetime(
            d.year, d.month, d.day, t.hour, t.minute,
            tzinfo=ZoneInfo(recipient_tz),
        )
        return _fmt(rec_dt.astimezone(ZoneInfo(sender_tz)).time())
    except Exception:
        return _fmt(t)


# ---------------------------------------------------------------------------
# Main schedule builder
# ---------------------------------------------------------------------------

def build_month_schedule(
    n_prospects: int,
    senders: list[str],
    year: int,
    month: int,
    bank_holidays: set[date],
    window_start: time,   # recipient-local; used for UI validation only
    window_end: time,     # recipient-local; upper bound for all slots
    sender1_start: time,  # recipient-local start for sender 1
    per_sender_cap: int,
    recipient_tz: str,
    sender_tz: str,
    variance_min: int = 4,
    variance_max: int = 8,
) -> list[ScheduledSend]:
    """
    Build a full month send schedule for n_prospects contacts.

    Returns a list of ScheduledSend of length == n_prospects,
    ordered chronologically (by date, then by recipient-local send time).

    Raises SchedulerError on any constraint violation.
    """
    if not senders:
        raise SchedulerError("No senders configured.")
    if n_prospects <= 0:
        return []

    working_days = get_working_days(year, month, bank_holidays)
    daily_targets = compute_daily_targets(n_prospects, working_days)
    max_s = compute_max_senders_per_day(window_end, sender1_start, len(senders))

    if max_s == 0:
        raise SchedulerError(
            f"Sender 1 start time ({sender1_start}) is at or past the window end "
            f"({window_end}). No senders can be scheduled."
        )

    result: list[ScheduledSend] = []

    for day, day_target in zip(working_days, daily_targets):
        if day_target == 0:
            continue

        available = senders[:max_s]
        allocs = allocate_senders_for_day(day_target, len(available), per_sender_cap)

        occupied: set[int] = set()
        # Collect (minute_key, send) so we can sort within the day before appending
        day_items: list[tuple[int, ScheduledSend]] = []

        for si, (sender, alloc) in enumerate(zip(available, allocs)):
            if alloc == 0:
                continue

            # Recipient-local start for this sender
            s_start = (_dt(sender1_start) + timedelta(minutes=si * 60)).time()

            baseline = generate_baseline_slots(s_start, window_end, alloc)
            jittered, occupied = apply_variance_no_overlap(
                baseline, occupied, window_end, variance_min, variance_max,
            )

            for t in jittered:
                time_str = to_sender_time_str(day, t, recipient_tz, sender_tz)
                display = f"{day.strftime('%a %d/%m')} {time_str}"
                key = t.hour * 60 + t.minute
                day_items.append((key, ScheduledSend(day, display, sender)))

        day_items.sort(key=lambda x: x[0])
        result.extend(s for _, s in day_items)

    return result
