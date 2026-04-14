# Creating Custom Games and Curricula

This guide walks you through building a completely new domain for HDNA Workbench — your own tasks, your own features, your own difficulty curve. Everything plugs into the existing tools, adapters, and training infrastructure.

## The Three Things You Need

To create a Workbench-compatible game, you need:

1. **A task generator** — function that produces (input, correct_answer, features)
2. **A feature extractor** — converts your domain state into a numeric vector
3. **A curriculum** — levels with difficulty progression built using CurriculumBuilder

That's it. Once you have these, every Workbench tool works with your game automatically: Inspector, Decision Replay, Experiment Forge, Daemon Studio, Exporter.

## Example: Building a "Word Length" Game

Let's build a simple game from scratch. The task: given a word, classify it as short (1-4 letters), medium (5-7), or long (8+).

### Step 1: Define Your Task Generator

```python
import numpy as np

# Your domain vocabulary
WORDS = {
    "short": ["cat", "dog", "hi", "go", "run", "the", "a", "is", "it", "on",
              "up", "an", "no", "do", "my", "we", "he", "me", "so", "or"],
    "medium": ["hello", "world", "python", "brain", "seven", "light",
               "music", "ocean", "dream", "smile", "house", "river"],
    "long": ["elephant", "beautiful", "computer", "wonderful", "adventure",
             "telephone", "astronomy", "dangerous", "education", "butterfly"],
}

CLASSES = ["short", "medium", "long"]

def generate_word_task(level: int, rng: np.random.Generator):
    """
    Generate one task for the word length game.
    
    Args:
        level: difficulty (0=obvious, 1=tricky, 2=hard)
        rng: random number generator
    
    Returns:
        (word, correct_class_index, features, difficulty)
    """
    # Pick a random class and word
    class_name = rng.choice(CLASSES)
    word = rng.choice(WORDS[class_name])
    correct_idx = CLASSES.index(class_name)
    
    # At higher levels, add noise (misspellings, unusual words)
    if level >= 1:
        # Sometimes pick borderline words
        if rng.random() < 0.3:
            borderline = {"short": "quiz", "medium": "jumps", "long": "learning"}
            word = borderline.get(class_name, word)
    
    difficulty = 0.2 + level * 0.3
    return word, correct_idx, difficulty
```

### Step 2: Define Your Feature Extractor

The feature extractor converts your domain-specific state (a word) into a numeric vector that HDNA can process.

```python
def word_features(word: str) -> np.ndarray:
    """
    Extract a 16-dimensional feature vector from a word.
    
    Design your features to capture what matters for your task.
    More informative features = faster learning.
    """
    features = np.zeros(16)
    
    # Length features
    features[0] = len(word) / 20.0                    # normalized length
    features[1] = 1.0 if len(word) <= 4 else 0.0      # is_short
    features[2] = 1.0 if 5 <= len(word) <= 7 else 0.0 # is_medium
    features[3] = 1.0 if len(word) >= 8 else 0.0      # is_long
    
    # Character features
    vowels = sum(1 for c in word.lower() if c in 'aeiou')
    consonants = sum(1 for c in word.lower() if c.isalpha() and c not in 'aeiou')
    features[4] = vowels / max(1, len(word))           # vowel ratio
    features[5] = consonants / max(1, len(word))       # consonant ratio
    
    # First/last character (encoded as position in alphabet)
    features[6] = (ord(word[0].lower()) - ord('a')) / 26.0 if word else 0
    features[7] = (ord(word[-1].lower()) - ord('a')) / 26.0 if word else 0
    
    # Syllable estimate (rough: count vowel groups)
    syllables = 0
    prev_vowel = False
    for c in word.lower():
        is_vowel = c in 'aeiou'
        if is_vowel and not prev_vowel:
            syllables += 1
        prev_vowel = is_vowel
    features[8] = syllables / 5.0
    
    # Character variety
    features[9] = len(set(word.lower())) / max(1, len(word))
    
    # Has double letters?
    features[10] = 1.0 if any(word[i] == word[i+1] for i in range(len(word)-1)) else 0.0
    
    # Padding for future features
    features[11:] = 0.0
    
    return features
```

### Step 3: Build the Curriculum

```python
from workbench.core import CurriculumBuilder

def build_word_curriculum(tasks_per_level=30, seed=42):
    rng = np.random.default_rng(seed)
    
    builder = CurriculumBuilder(
        "Word Length Classification",
        "Classify words as short, medium, or long"
    )
    
    for level in range(3):
        level_name = ["Easy", "Tricky", "Hard"][level]
        prereqs = [level - 1] if level > 0 else []
        
        builder.level(
            f"Word Length {level_name}",
            difficulty=0.2 + level * 0.3,
            prerequisites=prereqs,
            mastery_threshold=0.90,
            description=f"Level {level + 1}: {level_name} word classification",
        )
        
        for t in range(tasks_per_level):
            word, correct_idx, difficulty = generate_word_task(level, rng)
            features = word_features(word)
            
            builder.task(
                task_id=f"word_L{level}_{t}",
                input_data={"word": word},
                expected=correct_idx,
                features=features,
                difficulty=difficulty,
                metadata={
                    "word": word,
                    "class": CLASSES[correct_idx],
                    "length": len(word),
                },
            )
    
    return builder.build()
```

### Step 4: Train and Evaluate

```python
from workbench.core import HDNANetwork, Brain

# Build
curriculum = build_word_curriculum()
net = HDNANetwork(input_dim=16, output_dim=3, hidden_dims=[12, 6])
brain = Brain(net, epsilon=0.5, epsilon_decay=0.99)
rng = np.random.default_rng(42)

# Train
for episode in range(300):
    result = curriculum.get_task(rng)
    if result is None:
        break
    level, task = result
    
    action = brain.select_action(task.features, rng)
    correct = (action == task.expected_output)
    reward = 1.0 if correct else -0.2
    
    brain.learn(task.features, action, reward, rng.random(16), done=False)
    level.record_attempt(correct)

# Check progress
for level in curriculum.levels:
    snap = level.snapshot()
    print(f"{snap['name']:30s}  {snap['mastery']:10s}  {snap['accuracy']:.1%}")
```

### Step 5: Use All Workbench Tools

Your custom game now works with everything:

```python
from workbench.adapters import HDNAAdapter
from workbench.tools import Inspector, DecisionReplay, DaemonStudio, Exporter

# Wrap in adapter
adapter = HDNAAdapter(network=net, brain=brain, name="Word Classifier")

# Inspect
inspector = Inspector(adapter)
inspector.print_summary()

# Replay a decision
replayer = DecisionReplay(adapter)
test_word = "elephant"
replayer.print_trace(input_data=word_features(test_word))

# Test daemons on your curriculum
studio = DaemonStudio()
my_daemon = studio.from_template("argmax", name="word_argmax", num_actions=3)
result = studio.test(my_daemon, curriculum, episodes=100)

# Export
exporter = Exporter("./word_results")
exporter.table(curriculum.snapshot(), "progress.csv")
exporter.summary_report(inspector, "report.txt")
```

## Building a Custom Daemon for Your Game

If your domain needs specialized reasoning:

```python
from workbench.core import Daemon, Proposal

class WordLengthDaemon(Daemon):
    """Specialized daemon that reasons about word length."""
    
    def reason(self, state, features, rng=None):
        # features[0] = normalized length
        # features[1] = is_short flag
        # features[2] = is_medium flag  
        # features[3] = is_long flag
        
        length_estimate = features[0] * 20  # denormalize
        
        if length_estimate <= 4:
            action, confidence = 0, 0.9  # short
        elif length_estimate <= 7:
            action, confidence = 1, 0.8  # medium
        else:
            action, confidence = 2, 0.9  # long
        
        return Proposal(
            action=action,
            confidence=confidence,
            reasoning=f"Estimated length ~{length_estimate:.0f} chars",
            source=self.name,
        )

# Register it
from workbench.core import Coordinator
coordinator = Coordinator()
coordinator.register(WordLengthDaemon("word_length", domain="text",
                                      description="Classifies by word length"))
```

## Feature Engineering Tips

Good features make the difference between a model that learns and one that doesn't.

**Do**:
- Normalize values to roughly [0, 1] range
- Include features that directly relate to the correct answer
- Add structural features (counts, ratios, patterns)
- Start with more features than you need — prune later

**Don't**:
- Use raw strings or objects as features (convert to numbers)
- Leave features unnormalized (a feature ranging 0-1000 will dominate one ranging 0-1)
- Overfit features to specific examples (make them generalizable)

**Feature dimensions by domain**:

| Domain | Suggested Dims | Key Features |
|--------|---------------|--------------|
| Text classification | 16-64 | Word counts, character stats, keyword flags |
| Grid/spatial | 32-64 | Color counts, symmetry, object stats, spatial distribution |
| Math | 16-32 | Operand magnitudes, operator type, answer properties |
| Game/RL | 20-50 | State features, distance metrics, resource counts |
| Time series | 16-32 | Rolling stats, trend, periodicity, anomaly flags |
| Image (simple) | 32-128 | Pixel stats, edge counts, color histogram, symmetry |

## Packaging Your Game

To share your custom game as an installable package:

```
my-game/
    my_game/
        __init__.py
        tasks.py          # task generators
        features.py       # feature extractors
        curriculum.py     # curriculum factory
        daemons.py        # optional: domain-specific daemons
    setup.py or pyproject.toml
```

In `curriculum.py`:
```python
def my_game_curriculum(**kwargs):
    """Factory function that returns a ready-to-use Curriculum."""
    # ... build and return ...
```

Users then:
```python
pip install my-game
from my_game import my_game_curriculum
curriculum = my_game_curriculum()
# Works with all Workbench tools automatically
```

## Real-World Game Ideas

Some domains that researchers might build:

- **Medical triage**: Classify symptoms by urgency
- **Code review**: Identify bug patterns in code snippets
- **Financial signals**: Classify market indicators
- **Music theory**: Identify chord progressions
- **Chemistry**: Predict reaction outcomes
- **Logic puzzles**: Sudoku, constraint satisfaction
- **Language learning**: Vocabulary, grammar, translation
- **Robotics**: Navigation decisions from sensor data
