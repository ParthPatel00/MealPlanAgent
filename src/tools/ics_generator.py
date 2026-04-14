"""
Tool: ics_generator

Generates a valid .ics (iCalendar) file from a list of cooking time blocks.
The file can be imported into Google Calendar, Apple Calendar, or Outlook.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from icalendar import Calendar, Event


def generate_ics(
    cooking_blocks: list[dict],
    calendar_name: str = "Meal Plan",
) -> bytes:
    """
    Build an .ics file from a list of cooking block dicts.

    Args:
        cooking_blocks: List of dicts, each with:
            - meal_name (str): Recipe name
            - day (str): Day of week, e.g. "Monday"
            - cook_hour (int): Start hour in 24h format (e.g. 18 = 6 pm)
            - duration_minutes (int): How long cooking takes
        calendar_name: Display name for the calendar.

    Returns:
        Raw bytes of the .ics file.
    """
    cal = Calendar()
    cal.add("prodid", "-//MealPlanAgent//EN")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", calendar_name)

    # Map day names to the next upcoming occurrence from today
    today = date.today()
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }

    def next_weekday(day_name: str) -> date:
        target = day_map.get(day_name.lower(), 0)
        days_ahead = (target - today.weekday()) % 7
        return today + timedelta(days=days_ahead)

    for block in cooking_blocks:
        meal_name = block.get("meal_name", "Meal")
        day_str = block.get("day", "Monday")
        cook_hour = int(block.get("cook_hour", 18))
        duration = int(block.get("duration_minutes", 30))

        event_date = next_weekday(day_str)
        start_dt = datetime(
            event_date.year, event_date.month, event_date.day,
            cook_hour, 0, 0, tzinfo=timezone.utc
        )
        end_dt = start_dt + timedelta(minutes=duration)

        event = Event()
        event.add("summary", f"Cook: {meal_name}")
        event.add("dtstart", start_dt)
        event.add("dtend", end_dt)
        event.add("description", f"Cooking time block for {meal_name} ({duration} min)")

        cal.add_component(event)

    return cal.to_ical()


def save_ics(cooking_blocks: list[dict], path: str = "meal_plan.ics") -> str:
    """Write the .ics file to disk and return the path."""
    ics_bytes = generate_ics(cooking_blocks)
    with open(path, "wb") as f:
        f.write(ics_bytes)
    return path
