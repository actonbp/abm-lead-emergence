# Base Task Validation Plan

## Goal
Verify that our information-sharing task produces the same emergence patterns we see in the plot:
- Cognitive & Identity: Strong emergence (ICC > 0.8)
- Interactionist: Moderate emergence (ICC ≈ 0.4)
- Base & Null: Weak/no emergence (ICC ≈ 0.2)

## Task Implementation

### 1. Base Task Structure
```python
class BaseTask:
    def __init__(self, n_agents):
        # True solution (what group needs to discover)
        self.true_solution = generate_random_pattern(size=10)
        
        # Current group understanding
        self.group_understanding = initialize_neutral()
        
        # Distribute information to agents (70% accurate, 30% misleading)
        self.agent_info = distribute_information(n_agents)
        
        # Track shared information
        self.shared_info = set()
        
        # Track performance
        self.performance_history = []

    def share_info(self, from_agent, to_agent, granted):
        """
        Core task mechanic: sharing = claiming leadership
        Returns success/failure and associated costs
        """
        # Get unshared information
        available_info = get_unshared_info(from_agent)
        if not available_info:
            return {"success": False}
            
        # Select piece to share
        info_piece = select_random(available_info)
        
        if granted:
            # Update group understanding
            old_understanding = self.group_understanding.copy()
            self.update_understanding(info_piece)
            
            # Calculate improvement
            improvement = self.evaluate_improvement(
                old_understanding,
                self.group_understanding
            )
            
            return {
                "success": True,
                "improvement": improvement,
                "moves_closer": improvement > 0
            }
        else:
            return {"success": False}

    def evaluate_solution(self):
        """
        Measure how close group is to truth
        """
        return calculate_similarity(
            self.group_understanding,
            self.true_solution
        )
```

### 2. Integration with Models
```python
class BaseLeadershipModel:
    def step(self):
        # 1. Select interaction pair
        agent1, agent2 = self.select_pair()
        
        # 2. Calculate base probabilities
        match_score = self.calculate_match(agent1, agent2)
        
        # 3. Make leadership claim through task
        claim_result = self.task.share_info(
            from_agent=agent1,
            to_agent=agent2,
            granted=self.decide_grant(match_score)
        )
        
        # 4. Update based on result
        self.update_agents(claim_result)
        
        # 5. Track metrics
        self.track_metrics()
```

## Expected Patterns

### 1. Leadership Structure (ICC)
```
Time 0-20:
- All models similar (ICC ≈ 0.2)
- Initial exploration phase
- No clear leadership

Time 20-50:
- Cognitive & Identity rise sharply (ICC → 0.8)
- Interactionist rises moderately (ICC → 0.4)
- Base & Null stay flat (ICC ≈ 0.2)
```

### 2. Hierarchy Strength (Gini)
```
Time 0-20:
- All models show weak hierarchy
- Similar low Gini coefficients

Time 20-50:
- Cognitive & Identity develop stronger hierarchy
- Interactionist shows moderate hierarchy
- Base & Null maintain weak hierarchy
```

### 3. Leadership Entropy
```
Time 0-20:
- High entropy (≈ 1.0) for all models
- Uniform distribution of leadership

Time 20-50:
- Cognitive & Identity: Sharp drop in entropy
- Interactionist: Moderate drop
- Base & Null: Small decrease
```

### 4. Rank Volatility
```
Time 0-20:
- High volatility in all models
- Frequent rank changes

Time 20-50:
- Cognitive & Identity: Low volatility
- Interactionist: Moderate volatility
- Base & Null: High volatility
```

## Validation Process

### 1. Run Base Simulations
```python
for model_type in ['Base', 'Interactionist', 'Cognitive', 'Identity', 'Null']:
    results = []
    for run in range(30):  # 30 runs per model
        model = create_model(model_type)
        task = BaseTask(n_agents=6)
        model.set_task(task)
        
        # Run for 50 steps
        for step in range(50):
            state = model.step()
            metrics = calculate_metrics(state)
            results.append(metrics)
    
    plot_results(results)
```

### 2. Key Metrics to Track
```python
def calculate_metrics(state):
    return {
        'icc': calculate_leadership_icc(state),
        'gini': calculate_hierarchy_strength(state),
        'entropy': calculate_leadership_entropy(state),
        'volatility': calculate_rank_volatility(state),
        'task_performance': state.task.evaluate_solution()
    }
```

### 3. Success Criteria
- Emergence patterns match plot
- Clear differentiation between models
- Task performance correlates with emergence
- Stable patterns across multiple runs

## Implementation Steps

1. **Basic Task**
   - Implement core information sharing
   - Test basic mechanics
   - Verify information flow

2. **Model Integration**
   - Add task to each model
   - Ensure clean interaction
   - Test basic functionality

3. **Metric Tracking**
   - Implement all metrics
   - Add logging
   - Create visualizations

4. **Pattern Validation**
   - Run multiple simulations
   - Compare to expected patterns
   - Verify stability

5. **Fine-tuning**
   - Adjust parameters if needed
   - Maintain theoretical alignment
   - Document any changes

## Next Steps

1. **Implementation**
   - Create BaseTask class
   - Add to models
   - Set up metrics

2. **Testing**
   - Run base simulations
   - Compare patterns
   - Verify emergence

3. **Documentation**
   - Record parameters
   - Note any adjustments
   - Update predictions 