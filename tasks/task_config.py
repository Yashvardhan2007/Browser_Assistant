# ============================================================
# PASTE THIS IN: tasks/task_config.py
# Smart URL builder — goes directly to results, no wasted steps!
# ============================================================

SEARCH_GOOGLE = {
    "task_type": "search",
    "url": "https://www.google.com",
    "instructions": "Search on Google and STOP once results show.",
    "success_condition": {"url_contains": "search?q="},
}

FILL_CONTACT_FORM = {
    "task_type": "form_fill",
    "url": "https://httpbin.org/forms/post",
    "instructions": "Fill out the form and submit it.",
    "success_condition": {"url_contains": "httpbin.org"},
}

NAVIGATE_TO_SITE = {
    "task_type": "navigate",
    "url": "https://www.google.com",
    "instructions": "Navigate to the target URL and STOP.",
    "success_condition": {"url_contains": ""},
}


def get_task_config(task_type: str, **kwargs) -> dict:
    if task_type == "search":
        query = kwargs.get("query", "Python programming")
        config = SEARCH_GOOGLE.copy()
        config["instructions"] = f"Search for '{query}' on Google. STOP once results appear."
        config["query"] = query
        return config

    elif task_type == "form_fill":
        form_data = kwargs.get("form_data", {})
        url = kwargs.get("url", "https://httpbin.org/forms/post")
        config = FILL_CONTACT_FORM.copy()
        config["url"] = url
        config["form_data"] = form_data
        config["instructions"] = f"Fill the form at {url} with {form_data} and submit. STOP after submitting."
        return config

    elif task_type == "navigate":
        target_url = kwargs.get("target_url", "https://github.com")
        config = NAVIGATE_TO_SITE.copy()
        config["url"] = target_url
        config["instructions"] = f"You are already at {target_url}. Task is complete. STOP."
        config["success_condition"]["url_contains"] = target_url.replace("https://", "").replace("http://", "").split("/")[0]
        return config

    elif task_type == "wikipedia":
        query = kwargs.get("query", "Reinforcement Learning")
        return {
            "task_type": "navigate",
            "url": f"https://en.wikipedia.org/wiki/Special:Search?search={query.replace(' ', '+')}",
            "instructions": f"You are on Wikipedia search for '{query}'. STOP immediately.",
            "success_condition": {"url_contains": "wikipedia.org"},
        }

    else:
        return {
            "task_type": "custom",
            "url": kwargs.get("url", "https://www.google.com"),
            "instructions": kwargs.get("instructions", "Complete the task and STOP."),
            "success_condition": kwargs.get("success_condition", {}),
        }


def parse_user_instruction(instruction: str) -> dict:
    """
    Parse natural language into a task config.
    Builds direct URLs wherever possible to avoid wasted steps!
    """
    instruction_lower = instruction.lower()
    query = ""

    # ── Helper to extract query ────────────────────────────
    def extract_query(text, remove_words):
        q = text.lower()
        for w in remove_words:
            q = q.replace(w, "")
        return q.strip()

    # ── YouTube ────────────────────────────────────────────
    if "youtube" in instruction_lower:
        query = extract_query(instruction_lower, [
            "search for", "search", "find", "look up", "play",
            "on youtube", "in youtube", "youtube"
        ])
        return {
            "task_type": "navigate",
            "url": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}",
            "instructions": f"You are on YouTube search results for '{query}'. STOP immediately.",
            "success_condition": {"url_contains": "youtube.com/results"},
        }

    # ── Wikipedia ─────────────────────────────────────────
    elif "wikipedia" in instruction_lower:
        query = extract_query(instruction_lower, [
            "search for", "search", "find", "look up",
            "on wikipedia", "in wikipedia", "wikipedia"
        ])
        return get_task_config("wikipedia", query=query or instruction)

    # ── Google search ──────────────────────────────────────
    elif any(w in instruction_lower for w in ["search", "google", "look up"]):
        query = extract_query(instruction_lower, [
            "search for", "search google for", "google for",
            "google", "search", "look up", "find",
            "on google", "in google"
        ])
        # Direct Google search URL — no navigation needed!
        return {
            "task_type": "navigate",
            "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "instructions": f"You are on Google search results for '{query}'. STOP immediately.",
            "success_condition": {"url_contains": "google.com/search"},
        }

    # ── Navigation ─────────────────────────────────────────
    elif any(w in instruction_lower for w in ["go to", "navigate", "open", "visit"]):
        target_url = instruction_lower
        for prefix in ["go to", "navigate to", "open", "visit"]:
            target_url = target_url.replace(prefix, "").strip()

        # Clean up URL
        target_url = target_url.strip().split(" ")[0]
        if not target_url.startswith("http"):
            target_url = "https://" + target_url

        return get_task_config("navigate", target_url=target_url)

    # ── Form fill ──────────────────────────────────────────
    elif any(w in instruction_lower for w in ["fill", "form", "submit"]):
        form_data = {}
        parts = instruction.split(",")
        for part in parts:
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip().split()[-1]
                form_data[key.strip()] = val.strip()

        url = "https://httpbin.org/forms/post"
        if "http" in instruction:
            for word in instruction.split():
                if word.startswith("http"):
                    url = word
                    break

        return get_task_config("form_fill", form_data=form_data, url=url)

    # ── Popular Indian sites ───────────────────────────────

    elif "irctc" in instruction_lower:

        return {

            "task_type": "navigate",

            "url": "https://www.irctc.co.in/nget/train-search",

            "instructions": "You are on IRCTC train booking page. STOP immediately, task is complete.",

            "success_condition": {"url_contains": "irctc.co.in"},

        }

    elif "flipkart" in instruction_lower:

        query = extract_query(instruction_lower, ["search", "find", "buy", "on flipkart", "flipkart"])

        return {

            "task_type": "navigate",

            "url": f"https://www.flipkart.com/search?q={query.replace(' ', '+')}",

            "instructions": f"You are on Flipkart search for '{query}'. STOP immediately.",

            "success_condition": {"url_contains": "flipkart.com/search"},

        }

    elif "amazon" in instruction_lower:

        query = extract_query(instruction_lower, ["search", "find", "buy", "on amazon", "amazon"])

        return {

            "task_type": "navigate",

            "url": f"https://www.amazon.in/s?k={query.replace(' ', '+')}",

            "instructions": f"You are on Amazon search for '{query}'. STOP immediately.",

            "success_condition": {"url_contains": "amazon.in/s"},

        }

    # ── Default: Google search ─────────────────────────────
    else:
        return {
            "task_type": "navigate",
            "url": f"https://www.google.com/search?q={instruction.replace(' ', '+')}",
            "instructions": f"You are on Google search results for '{instruction}'. STOP immediately.",
            "success_condition": {"url_contains": "google.com/search"},
        }