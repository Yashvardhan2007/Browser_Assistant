

import re


# ── Success condition detector ─────────────────────────────
def detect_final_goal(instruction: str) -> str:
    """
    Look at the LAST action in instruction and return
    what the final URL should contain.
    """
    instruction_lower = instruction.lower()

    # Tab names
    tab_match = re.findall(r"['\"]([^'\"]+)['\"][\s]+tab", instruction_lower)
    if tab_match:
        return tab_match[-1].replace(" ", "-").lower()

    # GitHub specific
    if "issues" in instruction_lower and "github" in instruction_lower:
        return "/issues"
    if "pull request" in instruction_lower or "/pulls" in instruction_lower:
        return "/pulls"
    if "releases" in instruction_lower and "github" in instruction_lower:
        return "/releases"
    if "wiki" in instruction_lower and "github" in instruction_lower:
        return "/wiki"
    if "actions" in instruction_lower and "github" in instruction_lower:
        return "/actions"
    if "commits" in instruction_lower and "github" in instruction_lower:
        return "/commits"

    # PyPI specific
    if "release history" in instruction_lower and "pypi" in instruction_lower:
        return "pypi.org/project"
    if "pypi" in instruction_lower:
        return "pypi.org"

    # Stack Overflow
    if "answer" in instruction_lower and "stackoverflow" in instruction_lower:
        return "stackoverflow.com/questions"
    if "stackoverflow" in instruction_lower:
        return "stackoverflow.com"

    # Reddit
    if "comment" in instruction_lower and "reddit" in instruction_lower:
        return "reddit.com/r"
    if "reddit" in instruction_lower:
        return "reddit.com"

    # Wikipedia
    if "wikipedia" in instruction_lower:
        return "wikipedia.org/wiki/"

    # Arxiv
    if "arxiv" in instruction_lower:
        return "arxiv.org"

    # Kaggle
    if "kaggle" in instruction_lower:
        return "kaggle.com"

    return ""


# ── Website detector ───────────────────────────────────────
def detect_website(instruction: str) -> str:
    """Detect if instruction mentions a specific website."""
    instruction_lower = instruction.lower()

    KNOWN_SITES = {
        "news.ycombinator.com": "https://news.ycombinator.com",
        "hacker news": "https://news.ycombinator.com",
        "hackernews": "https://news.ycombinator.com",
        "ycombinator": "https://news.ycombinator.com",
        "pypi.org": "https://pypi.org",
        "pypi": "https://pypi.org",
        "github.com/search": "https://github.com/search",
        "github.com": "https://github.com",
        "github": "https://github.com",
        "stackoverflow.com": "https://stackoverflow.com",
        "stackoverflow": "https://stackoverflow.com",
        "stack overflow": "https://stackoverflow.com",
        "huggingface.co": "https://huggingface.co",
        "huggingface": "https://huggingface.co",
        "hugging face": "https://huggingface.co",
        "pytorch.org": "https://pytorch.org",
        "pytorch": "https://pytorch.org",
        "arxiv.org": "https://arxiv.org",
        "arxiv": "https://arxiv.org",
        "kaggle.com": "https://kaggle.com",
        "kaggle": "https://kaggle.com",
        "npmjs.com": "https://www.npmjs.com",
        "npm": "https://www.npmjs.com",
        "reddit.com": "https://www.reddit.com",
        "reddit": "https://www.reddit.com",
        "medium.com": "https://medium.com",
        "medium": "https://medium.com",
        "python.org": "https://www.python.org",
        "docs.python.org": "https://docs.python.org",
        "linkedin.com": "https://www.linkedin.com",
        "linkedin": "https://www.linkedin.com",
        "twitter.com": "https://twitter.com",
        "x.com": "https://x.com",
    }

    # Check longest match first to avoid partial matches
    for keyword in sorted(KNOWN_SITES.keys(), key=len, reverse=True):
        if keyword in instruction_lower:
            return KNOWN_SITES[keyword]

    # Check for any domain pattern
    domain_pattern = re.findall(
        r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.(?:com|org|io|dev|net|co|in|ai|gov|edu)(?:/[a-zA-Z0-9-/]*)?)',
        instruction
    )
    if domain_pattern:
        domain = domain_pattern[0]
        if not domain.startswith("http"):
            return "https://" + domain
        return domain

    return ""


# ── Query extractor ────────────────────────────────────────
def extract_query(text: str, remove_words: list) -> str:
    """Extract just the search query — remove all instruction words."""
    q = text.lower()
    for w in sorted(remove_words, key=len, reverse=True):
        q = q.replace(w, "")
    return q.strip().strip("'\".,!?()")


# ── Main get_task_config ───────────────────────────────────
def get_task_config(task_type: str, **kwargs) -> dict:
    if task_type == "search":
        query = kwargs.get("query", "Python programming")
        return {
            "task_type": "search",
            "url": "https://www.google.com",
            "instructions": f"Search for '{query}' on Google. STOP once results appear.",
            "query": query,
            "success_condition": {"url_contains": "search?q="},
        }
    elif task_type == "form_fill":
        form_data = kwargs.get("form_data", {})
        url = kwargs.get("url", "https://httpbin.org/forms/post")
        return {
            "task_type": "form_fill",
            "url": url,
            "form_data": form_data,
            "instructions": f"Fill the form at {url} with {form_data} and submit. STOP after submitting.",
            "success_condition": {"url_contains": "httpbin.org"},
        }
    elif task_type == "navigate":
        target_url = kwargs.get("target_url", "https://github.com")
        clean = target_url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
        return {
            "task_type": "navigate",
            "url": target_url,
            "instructions": f"You are at {target_url}. STOP immediately.",
            "success_condition": {"url_contains": clean},
        }
    else:
        return {
            "task_type": "custom",
            "url": kwargs.get("url", "https://www.google.com"),
            "instructions": kwargs.get("instructions", "Complete the task and STOP."),
            "success_condition": kwargs.get("success_condition", {}),
        }


# ── MAIN PARSER ────────────────────────────────────────────
def parse_user_instruction(instruction: str) -> dict:
    """
    SMART parser that handles ALL instruction types:
    - Simple: "search X on Google"
    - Navigate: "go to github.com"  
    - Multi-step: "go to pypi.org, search playwright, click Issues tab"
    - Complex: "find #1 story on HN, search it on Google"
    """
    instruction_lower = instruction.lower()

    # ══════════════════════════════════════════════════════
    # PRIORITY 1: Detect specific website
    # Always check this first — most important!
    # ══════════════════════════════════════════════════════
    website_url = detect_website(instruction)

    if website_url:
        # Detect what the FINAL goal URL should contain
        final_goal = detect_final_goal(instruction)
        domain = website_url.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]

        # If no final goal detected, use domain as fallback
        success_url = final_goal if final_goal else domain

        return {
            "task_type": "multi_step",
            "url": website_url,
            "instructions": f"""Your task: {instruction}

You are starting at: {website_url}
Complete the task step by step.
Read the page carefully before each action.
Take ONE action at a time.
STOP only when you have fully completed the FINAL goal of the task.
The final goal is reached when the URL contains: '{success_url}'""",
            "success_condition": {"url_contains": success_url},
        }

    # ══════════════════════════════════════════════════════
    # PRIORITY 2: YouTube
    # ══════════════════════════════════════════════════════
    elif "youtube" in instruction_lower:
        query = extract_query(instruction_lower, [
            "search for", "search", "find", "look up", "play", "watch",
            "on youtube", "in youtube", "youtube", "video about", "videos about",
            "video of", "videos of", "show me"
        ])
        if not query:
            return get_task_config("navigate", target_url="https://www.youtube.com")
        return {
            "task_type": "navigate",
            "url": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}",
            "instructions": f"You are on YouTube results for '{query}'. STOP immediately.",
            "success_condition": {"url_contains": "youtube.com/results"},
        }

    # ══════════════════════════════════════════════════════
    # PRIORITY 3: Wikipedia
    # ══════════════════════════════════════════════════════
    elif "wikipedia" in instruction_lower:
        query = extract_query(instruction_lower, [
            "search for", "search", "find", "look up", "go to", "open",
            "on wikipedia", "in wikipedia", "wikipedia", "article about",
            "article on", "page about", "page on", "the", "a"
        ])
        if not query:
            return get_task_config("navigate", target_url="https://www.wikipedia.org")
        return {
            "task_type": "navigate",
            "url": f"https://en.wikipedia.org/wiki/Special:Search?search={query.replace(' ', '+')}",
            "instructions": f"You are on Wikipedia search for '{query}'. STOP immediately.",
            "success_condition": {"url_contains": "wikipedia.org/wiki/"},
        }

    # ══════════════════════════════════════════════════════
    # PRIORITY 4: Google search
    # ══════════════════════════════════════════════════════
    elif any(w in instruction_lower for w in ["search for", "search", "google", "look up"]):
        query = extract_query(instruction_lower, [
            "search for", "search google for", "google for", "google",
            "search", "look up", "find", "on google", "in google",
            "please", "can you", "i want to", "i need to", "help me",
            "show me", "tell me about", "what is", "how to"
        ])
        if not query:
            return get_task_config("navigate", target_url="https://www.google.com")
        return {
            "task_type": "navigate",
            "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "instructions": f"You are on Google results for '{query}'. STOP immediately.",
            "success_condition": {"url_contains": "google.com/search"},
        }

    # ══════════════════════════════════════════════════════
    # PRIORITY 5: Popular Indian sites
    # ══════════════════════════════════════════════════════
    elif "irctc" in instruction_lower:
        return {
            "task_type": "navigate",
            "url": "https://www.irctc.co.in/nget/train-search",
            "instructions": "You are on IRCTC train search. STOP immediately.",
            "success_condition": {"url_contains": "irctc.co.in"},
        }

    elif "flipkart" in instruction_lower:
        query = extract_query(instruction_lower, [
            "search", "find", "buy", "on flipkart", "flipkart", "for", "get"
        ])
        return {
            "task_type": "navigate",
            "url": f"https://www.flipkart.com/search?q={query.replace(' ', '+')}",
            "instructions": f"You are on Flipkart search for '{query}'. STOP immediately.",
            "success_condition": {"url_contains": "flipkart.com/search"},
        }

    elif "amazon" in instruction_lower:
        query = extract_query(instruction_lower, [
            "search", "find", "buy", "on amazon", "amazon", "for", "get"
        ])
        return {
            "task_type": "navigate",
            "url": f"https://www.amazon.in/s?k={query.replace(' ', '+')}",
            "instructions": f"You are on Amazon search for '{query}'. STOP immediately.",
            "success_condition": {"url_contains": "amazon.in/s"},
        }

    # ══════════════════════════════════════════════════════
    # PRIORITY 6: Simple navigation
    # ══════════════════════════════════════════════════════
    elif any(w in instruction_lower for w in ["go to", "navigate to", "open", "visit"]):
        target_url = instruction_lower
        for prefix in ["go to", "navigate to", "open", "visit", "please", "can you"]:
            target_url = target_url.replace(prefix, "").strip()
        target_url = target_url.strip().split(" ")[0].strip("'\".,")
        if not target_url.startswith("http"):
            target_url = "https://" + target_url
        return get_task_config("navigate", target_url=target_url)

    # ══════════════════════════════════════════════════════
    # PRIORITY 7: Form fill
    # ══════════════════════════════════════════════════════
    elif any(w in instruction_lower for w in ["fill", "form", "submit"]):
        form_data = {}
        for part in instruction.split(","):
            if "=" in part:
                key, val = part.split("=", 1)
                form_data[key.strip().split()[-1]] = val.strip()
        url = "https://httpbin.org/forms/post"
        for word in instruction.split():
            if word.startswith("http"):
                url = word
                break
        return get_task_config("form_fill", form_data=form_data, url=url)

    # ══════════════════════════════════════════════════════
    # PRIORITY 8: Default — clean Google search
    # ══════════════════════════════════════════════════════
    else:
        query = extract_query(instruction_lower, [
            "find", "what is", "how to", "tell me about", "show me",
            "i want to know", "can you", "please", "what are", "who is"
        ])
        if not query:
            query = instruction
        return {
            "task_type": "navigate",
            "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
            "instructions": f"You are on Google results for '{query}'. STOP immediately.",
            "success_condition": {"url_contains": "google.com/search"},
        }