class PromptTemplates:
    ACTION_GUESS_PROMPT = (
        "Reason very very succinctly about the {i}th action, return {num_guesses} candidates for the next action. IMPORTANT FORMAT: "
        "Your response must be a first a thought based on the current situation and then the {num_guesses} options for actions as a list."
        "(exact syntax with square brackets and commas between actions!). For example: Action {i}: Search[Barack Obama], Search[Obama], Lookup[Barack] "
        "Ensure all actions are from the list of actions - Search, Lookup and Finish. Even if uncertain, return exactly {num_guesses} candidates. "
        "Reason very very quickly, no need to truly think about each of the actions, just return the most likely candidates and return precisely {num_guesses} valid actions - no more, no less."
    )

    RETRY_PROMPT = (
        "Attempt {attempt} failed, please remember that you are acting as {role}"
        "and you need to output {num_guesses} valid actions out of the three possible actions - Search, Lookup and Finish. "
    )

    REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types:
        (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
        (2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
        (3) Finish[answer], which returns the answer and finishes the task.
        Here are some examples.
        """

    PROMPT_INSTRUCTION = """Now answer the question. Start with the thought always. Do not under any circumstances give me the actions first."""

    NEXT_STEP_PROMPT = """Thought {i}: {thought}
        Action {i}: {action}
        Observation {i}: {obs}"
        """

    GUESS_STEP_PROMPT = "Give me information about {} in wikipedia style. If wikipedia information is not available still give me the information. Do not give me any error messages. Just give me the information and nothing else"

    MEMORY_HINT_PROMPT = (
        "## Action Prediction Guide\n\n"
        "{insight_section}"
        "{example_section}"
        "Use these patterns and examples to inform your action candidates.\n"
    )

    INSIGHT_EXTRACTION_PROMPT = (
        "You are analyzing action prediction patterns from a speculative execution agent "
        "that uses Search, Lookup, and Finish actions to answer questions using Wikipedia.\n\n"
        "## Recent Predictions (+ = success, - = failure):\n"
        "{raw_entries}\n\n"
        "## Existing Insights (avoid duplicates; refine if you have more data):\n"
        "{existing_insights}\n\n"
        "## Task\n"
        "Analyze the above predictions and extract 3-5 concise, actionable insights "
        "written in natural English (one or two sentences each). Each insight should:\n"
        "1. Describe a recurring pattern in action selection\n"
        "2. Be directly useful for guiding future action predictions\n"
        "3. Be non-obvious (not just \"step 1 is always Search\")\n\n"
        "For each insight, also provide 3-8 keywords that would help match this insight "
        "to relevant questions.\n\n"
        "Return ONLY a JSON array. Each element must have exactly two fields:\n"
        '- "insight": the natural language insight (1-2 sentences)\n'
        '- "keywords": array of relevant trigger words\n\n'
        "No other text."
    )
