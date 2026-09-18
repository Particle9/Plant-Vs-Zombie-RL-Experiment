# Heuristic AI Agent for Plants vs Zombies

## Overview

The `HeuristicAgent` is a rule-based AI that makes decisions based on the current game state using hand-crafted heuristics rather than learning from data. This agent can serve as a meaningful baseline for comparison against reinforcement learning models.

## Strategy

The agent implements a multi-level priority system:

### 1. **Emergency Defense** (Highest Priority)
- Monitors threat level on each lane (based on zombie proximity and health)
- Places defensive plants (Wallnut, Chomper) on high-threat lanes
- Triggered when threat score > 3.0

### 2. **Medium Threat Response** (Second Priority)
- Places offensive plants (Repeater, Peashooter) on lanes with moderate threats
- Requires at least 100 sun to maintain economic balance
- Triggered when threat score between 1.0 and 3.0

### 3. **Sun Generation** (Third Priority)
- Places Sunflowers on safe lanes when sun is low (< 150)
- Focuses on lanes with minimal zombie threat
- Ensures economy doesn't bottleneck

### 4. **Offensive Pressure** (Fourth Priority)
- Places additional offensive plants when sun is available
- Attempts to kill zombies faster to prevent lane overwhelm

## Threat Analysis

Each lane receives a threat score calculated as:
```
threat_score = total_zombie_hp / (1 + distance_to_closest_zombie)
```

This prioritizes:
- Lanes with closer zombies
- Lanes with more total zombie health
- Creates urgency for defense when zombies approach the lawn

## Plant Selection Logic

### Plant Types
1. **Sunflower** (ID: 0) - Sun generation
2. **Peashooter** (ID: 1) - Basic offense
3. **Wallnut** (ID: 2) - Defense/blocking
4. **Chomper** (ID: 3) - Close-range damage
5. **Repeater** (ID: 4) - Enhanced offense
6. **Jalapeno** (ID: 5) - Area damage (reserved for future enhancement)

### Placement Strategy
- Prefers positioning plants toward the back of lanes (safe zone)
- Focuses on empty cells first
- Validates all placements against action mask

## Usage

### Import
```python
from agents import HeuristicAgent

# Create agent
agent = HeuristicAgent(possible_actions=env.get_actions())

# Use in game
action = agent.decide_action(observation, mask, epsilon=0)
```

### Integration with Comparison Notebook
The agent is automatically included in `comparison.ipynb`:
- Runs 100 episodes alongside other agents
- Reported in summary statistics (score, ticks survived, waves reached)
- Visualized in comparison charts

## Expected Performance

As a rule-based system, the Heuristic Agent should:
- ✅ Outperform random and do-nothing baselines
- ✅ Provide a reasonable middle ground strategy
- ❓ Compete with or potentially underperform trained RL models
- ✓ Run instantly with no model loading time

## Advantages
- **Interpretability**: Clear decision logic for every action
- **Speed**: No neural network inference required
- **Baseline**: Good sanity check for RL agent performance
- **Debugging**: Easy to adjust weights and strategies

## Limitations
- **Suboptimal**: Hand-crafted rules can't match learned policies
- **Static**: Doesn't adapt to new game situations
- **Limited Strategies**: Can't discover emergent tactics like RL agents
- **No Learning**: Repeats same strategy regardless of success

## Future Improvements

Potential enhancements:
1. Add Jalapeno emergency crowd-control logic
2. Implement lane prioritization based on history
3. Add dynamic threat threshold adjustment
4. Consider plant synergies and combinations
5. Implement seasonal strategy changes (early vs late game)

## Files Modified/Created

- **New**: `/agents/heuristic_agent.py` - Main agent implementation
- **Modified**: `/agents/__init__.py` - Added HeuristicAgent export
- **Modified**: `/comparison.ipynb` - Added Heuristic agent spec and import

## Testing

To test the heuristic agent:
```python
from agents import PlayerQ, HeuristicAgent

env = PlayerQ(render=False, max_waves=100)
agent = HeuristicAgent(possible_actions=env.get_actions())

# Play one episode
summary = env.play(agent)
print(f"Score: {sum(summary['raw_rewards'])}")
print(f"Waves: {summary['observations'].shape[0]}")
```
