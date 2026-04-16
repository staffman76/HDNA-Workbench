# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
Language Curriculum — Sentiment, topic, emotion, and intent classification.

Extracted from HDNA-LM which achieved 97-100% accuracy across all 4 tasks
with zero catastrophic forgetting. Uses template-based text with word-level
features.

Each task has 3 difficulty levels:
    L1: Obvious markers ("I love this" = positive)
    L2: Contextual ("Not bad at all" = positive via negation)
    L3: Subtle/mixed ("The irony wasn't lost on anyone" = nuanced)

Features are bag-of-words vectors with vocabulary fitted from all templates.
"""

import numpy as np
from ..core.curriculum import Curriculum, CurriculumBuilder, Task


# ============================================================
# Text templates per task and level
# ============================================================

SENTIMENT = {
    "L1": {
        "positive": [
            "this is great", "i love this", "absolutely wonderful",
            "best thing ever", "really amazing work", "so happy with this",
            "fantastic results", "excellent quality", "truly impressive",
            "outstanding performance", "love it so much", "perfect solution",
            "brilliant idea", "superb execution", "magnificent effort",
            "delightful experience", "pure joy", "thrilled with the outcome",
            "incredibly satisfying", "remarkable achievement",
        ],
        "negative": [
            "this is terrible", "i hate this", "absolutely awful",
            "worst thing ever", "really disappointing", "so frustrated",
            "horrible results", "poor quality", "truly unacceptable",
            "pathetic performance", "hate it completely", "terrible solution",
            "stupid idea", "awful execution", "miserable failure",
            "dreadful experience", "pure misery", "devastated by the outcome",
        ],
        "neutral": [
            "it is what it is", "the meeting is at three", "the report was filed",
            "temperature is seventy two", "the package arrived today",
            "the update was installed", "data was collected", "the schedule is set",
            "information was recorded", "the system is running",
        ],
    },
    "L2": {
        "positive": [
            "not bad at all actually", "better than i expected",
            "surprisingly decent work", "turned out well in the end",
            "exceeded my low expectations", "pleasantly surprised by this",
            "grew on me over time", "not what i feared it would be",
        ],
        "negative": [
            "not as good as i hoped", "could have been much better",
            "left a lot to be desired", "fell short of expectations",
            "not the worst but close", "barely acceptable at best",
            "mediocre would be generous", "underwhelming to say the least",
        ],
        "neutral": [
            "it performed as specified", "results were within range",
            "nothing unexpected occurred", "standard operating procedure",
            "metrics aligned with projections", "baseline was maintained",
        ],
    },
    "L3": {
        "positive": [
            "the subtle craftsmanship speaks volumes",
            "a quiet confidence permeates the work",
            "understated elegance at its finest",
            "the attention to nuance is refreshing",
        ],
        "negative": [
            "the silence after was deafening",
            "a masterclass in missing the point",
            "one wonders who approved this direction",
            "the gap between ambition and execution is vast",
        ],
        "neutral": [
            "the data neither confirms nor denies the hypothesis",
            "an artifact of the methodology perhaps",
            "the correlation remains statistically insignificant",
        ],
    },
}

TOPIC = {
    "L1": {
        "sports": [
            "the team won the championship game", "scored three goals in the match",
            "the quarterback threw a touchdown", "ran a marathon in under three hours",
            "the basketball playoffs start next week", "swimming records were broken",
            "the tennis tournament concluded today", "boxing match ended in knockout",
        ],
        "technology": [
            "the new processor runs faster", "software update fixes the bug",
            "machine learning improves accuracy", "the server was upgraded yesterday",
            "cloud computing reduces costs", "the algorithm optimizes performance",
            "database migration completed successfully", "api response time decreased",
        ],
        "food": [
            "the pasta was perfectly al dente", "fresh herbs enhance the flavor",
            "baked the bread from scratch", "the recipe calls for two eggs",
            "grilled vegetables taste amazing", "chocolate cake for dessert",
            "the soup needs more seasoning", "organic ingredients are preferred",
        ],
        "science": [
            "the experiment confirmed the theory", "molecules bonded under pressure",
            "the telescope detected a new signal", "dna sequencing revealed mutations",
            "gravity affects all objects equally", "the chemical reaction was exothermic",
            "photosynthesis converts light to energy", "the vaccine targets the protein",
        ],
        "health": [
            "exercise improves cardiovascular health", "the patient recovered fully",
            "vitamin d supports bone density", "blood pressure was within normal range",
            "meditation reduces stress levels", "the diagnosis was confirmed by tests",
            "healthy diet prevents many diseases", "sleep quality affects cognition",
        ],
        "finance": [
            "stock prices rose three percent", "the interest rate was lowered",
            "quarterly earnings exceeded projections", "inflation affects purchasing power",
            "the portfolio was diversified", "mortgage rates hit record lows",
            "revenue grew year over year", "the budget was approved by the board",
        ],
    },
}

EMOTION = {
    "L1": {
        "happy": [
            "i am so happy today", "this makes me smile", "feeling wonderful",
            "joy fills my heart", "what a beautiful day", "life is amazing",
            "everything is going perfectly", "i could not be more pleased",
        ],
        "sad": [
            "i feel so lonely", "this makes me cry", "feeling down today",
            "my heart is heavy", "nothing seems right", "i miss them so much",
            "everything feels hopeless", "tears keep falling",
        ],
        "angry": [
            "i am furious about this", "this makes me so mad",
            "how dare they do this", "absolutely infuriating behavior",
            "i cannot believe this nonsense", "rage is all i feel",
            "this is completely unacceptable", "beyond frustrated right now",
        ],
        "scared": [
            "i am terrified of what comes next", "this is frightening",
            "my hands are shaking", "fear grips me completely",
            "i dread the outcome", "the uncertainty is paralyzing",
            "something feels very wrong", "panic is setting in",
        ],
        "surprised": [
            "i did not see that coming", "what a shock this is",
            "completely unexpected turn of events", "i cannot believe my eyes",
            "this is so surprising", "never would have predicted this",
            "jaw dropping revelation", "stunned by the announcement",
        ],
    },
}

INTENT = {
    "L1": {
        "request": [
            "can you help me with this", "please send the report",
            "would you mind checking", "i need assistance with the task",
            "could you look into this issue", "please review my submission",
            "help me understand this better", "i would appreciate your input",
        ],
        "inform": [
            "the meeting has been moved to friday", "i wanted to let you know",
            "for your information the project shipped", "just an update on progress",
            "the results are now available", "fyi the deadline changed",
            "here is the latest status", "reporting that the task is complete",
        ],
        "question": [
            "what time does the meeting start", "how do i access the system",
            "where can i find the documentation", "who is responsible for this",
            "when is the deadline", "why was the decision made",
            "which option should i choose", "is this the correct approach",
        ],
        "command": [
            "stop the process immediately", "restart the server now",
            "delete the old files", "run the test suite",
            "deploy to production", "update the configuration",
            "cancel the subscription", "execute the migration",
        ],
        "agree": [
            "yes that sounds right", "i agree with your assessment",
            "absolutely correct", "that makes perfect sense",
            "you are right about that", "exactly what i was thinking",
            "i support this decision", "confirmed and approved",
        ],
        "refuse": [
            "no i cannot do that", "i disagree with this approach",
            "that is not acceptable", "i decline the offer",
            "this will not work for us", "absolutely not",
            "i object to this proposal", "we cannot proceed this way",
        ],
    },
}


# ============================================================
# Vocabulary & Feature extraction
# ============================================================

def _build_vocabulary(task_data: dict) -> dict:
    """Build a word-to-index vocabulary from all templates."""
    words = set()
    for levels in task_data.values():
        for label, texts in levels.items():
            for text in texts:
                words.update(text.lower().split())

    vocab = {"<pad>": 0, "<unk>": 1}
    for i, word in enumerate(sorted(words)):
        vocab[word] = i + 2
    return vocab


def _text_to_features(text: str, vocab: dict, max_len: int = 50) -> np.ndarray:
    """
    Convert text to a bag-of-words feature vector.

    Each dimension corresponds to a vocabulary word. Value is
    1.0 if the word is present, 0.0 otherwise.
    """
    features = np.zeros(len(vocab))
    for word in text.lower().split():
        idx = vocab.get(word, 1)  # 1 = <unk>
        features[idx] = 1.0
    return features


# ============================================================
# Curriculum factory
# ============================================================

def _build_task_curriculum(builder, task_name: str, task_data: dict,
                           level_offset: int, vocab: dict,
                           tasks_per_level: int, rng: np.random.Generator):
    """Build levels for one language task."""
    labels = None
    level_id = level_offset

    for level_name, level_data in task_data.items():
        if labels is None:
            labels = sorted(level_data.keys())
        label_to_idx = {label: i for i, label in enumerate(labels)}

        difficulty = {"L1": 0.2, "L2": 0.5, "L3": 0.8}.get(level_name, 0.5)
        prereqs = [level_id - 1] if level_id > level_offset else []

        builder.level(
            f"{task_name} {level_name}",
            difficulty=difficulty,
            prerequisites=prereqs,
            mastery_threshold=0.55,
            description=f"{task_name}: {level_name} difficulty, {len(labels)} classes",
            tags={task_name.lower()},
        )

        # Generate tasks from templates
        task_count = 0
        for label, texts in level_data.items():
            for text in texts:
                if task_count >= tasks_per_level:
                    break
                features = _text_to_features(text, vocab)
                builder.task(
                    task_id=f"lang_{task_name.lower()}_{level_name}_{task_count}",
                    input_data={"text": text, "task": task_name},
                    expected=label_to_idx[label],
                    features=features,
                    difficulty=difficulty,
                    tags={task_name.lower(), label},
                    metadata={
                        "text": text,
                        "label": label,
                        "label_idx": label_to_idx[label],
                        "task": task_name,
                        "level": level_name,
                        "all_labels": labels,
                    },
                )
                task_count += 1

        level_id += 1

    return level_id


def language_curriculum(tasks: list = None, tasks_per_level: int = 50,
                        seed: int = 42) -> Curriculum:
    """
    Build the language curriculum.

    Args:
        tasks: Which tasks to include. Default: all 4.
                Options: ["sentiment", "topic", "emotion", "intent"]
        tasks_per_level: Max tasks per difficulty level.
        seed: Random seed.

    Returns:
        A Curriculum with text classification tasks across multiple
        domains, bag-of-words features, and 3 difficulty levels per task.
    """
    rng = np.random.default_rng(seed)

    available = {
        "sentiment": SENTIMENT,
        "topic": TOPIC,
        "emotion": EMOTION,
        "intent": INTENT,
    }

    if tasks is None:
        tasks = list(available.keys())

    # Build unified vocabulary
    all_data = {}
    for name in tasks:
        if name in available:
            all_data[name] = available[name]

    combined = {}
    for name, data in all_data.items():
        combined.update(data)
    vocab = _build_vocabulary({"all": combined})

    builder = CurriculumBuilder(
        "Language",
        f"{len(tasks)}-task language curriculum: {', '.join(tasks)}"
    )

    level_offset = 0
    for task_name in tasks:
        if task_name in all_data:
            level_offset = _build_task_curriculum(
                builder, task_name.capitalize(), all_data[task_name],
                level_offset, vocab, tasks_per_level, rng
            )

    return builder.build()
