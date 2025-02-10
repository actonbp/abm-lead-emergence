# Puzzle Task Validation

## Simple Validation Run
```python
def run_puzzle_validation(n_steps=50):
    """
    Run a single simulation with PuzzleTask to verify mechanics.
    """
    # Initialize with base model's best parameters
    params = {
        'n_agents': 6,
        'match_threshold': 0.41,
        'base_claim_probability': 0.7
    }
    
    # Create model and puzzle task
    model = BaseModel(params)
    task = PuzzleTask(
        num_agents=params['n_agents'],
        puzzle_size=10,  # 10 pieces total
        num_stages=3     # 3 stages to complete
    )
    model.set_task(task)
    
    # Track key metrics
    stage_logs = []
    resource_logs = []
    time_logs = []
    
    print("Starting puzzle validation run...")
    print("\nInitial State:")
    print_puzzle_state(task)
    
    # Run simulation with detailed logging
    for step in range(n_steps):
        state = model.step()
        
        # Log metrics
        stage_logs.append({
            'current_stage': task.current_stage,
            'stage_quality': task._evaluate_stage_understanding(),
            'overall_quality': task._evaluate_understanding()
        })
        
        resource_logs.append({
            'resources': task.resources,
            'time_spent': task.time_spent
        })
        
        # Print detailed info every 10 steps
        if step % 10 == 0:
            print(f"\nStep {step}:")
            print_puzzle_state(task)
            print_recent_interactions(state)
    
    return {
        'stage_logs': stage_logs,
        'resource_logs': resource_logs,
        'time_logs': time_logs
    }

def print_puzzle_state(task):
    """Print current state of puzzle task."""
    print(f"Current Stage: {task.current_stage + 1}/{task.num_stages}")
    print(f"Resources Remaining: {task.resources:.1f}/{task.initial_resources}")
    print(f"Time Spent: {task.time_spent}")
    
    print("\nStage Progress:")
    for stage in range(task.num_stages):
        quality = np.mean(np.abs(
            task.current_understanding[stage] - 
            task.true_solutions[stage]
        ))
        print(f"Stage {stage + 1}: {(1-quality)*100:.1f}% complete")
        
        if stage in task.stage_leaders:
            print(f"  Led by Agent {task.stage_leaders[stage]}")
    
    print("\nShared Information:")
    shared_by_stage = {}
    for stage, piece in task.shared_pieces:
        shared_by_stage[stage] = shared_by_stage.get(stage, 0) + 1
    
    for stage in range(task.num_stages):
        pieces_shared = shared_by_stage.get(stage, 0)
        total_pieces = len(task.true_solutions[stage])
        print(f"Stage {stage + 1}: {pieces_shared}/{total_pieces} pieces shared")

def print_recent_interactions(state):
    """Print details of recent interactions."""
    print("\nRecent Interactions:")
    for interaction in state['recent_interactions']:
        print(f"Agent {interaction['claimer']} → Agent {interaction['target']}:")
        print(f"- Success: {interaction['success']}")
        if interaction['success']:
            print(f"- Stage: {interaction['shared_info'][0] + 1}")
            print(f"- Quality: {interaction['quality']:.3f}")
            print(f"- Cost: {interaction['cost']:.1f}")
            print(f"- Time: {interaction['time']:.1f}")
```

## Expected Patterns

### 1. Stage Progression
```
Should see:
- Stages complete sequentially
- Quality improves within each stage
- Leaders emerge for each stage
- Clear stage transitions
```

### 2. Resource Management
```
Expected pattern:
- Resources decrease over time
- Leaders use resources more efficiently
- Failed claims cost less
- Resource use affects behavior
```

### 3. Time Management
```
Should observe:
- Time accumulates with actions
- Leaders act more quickly
- Stage completion times vary
- Overall progress reasonable
```

### 4. Information Flow
```
Should see:
- Information shared within stages
- Quality improves over time
- Leaders emerge from good sharing
- Clear stage completion criteria
```

## Success Criteria

### 1. Stage Mechanics
- Stages complete in order
- Quality improves within stages
- Leaders emerge naturally
- Clear stage transitions

### 2. Resource Mechanics
- Resource usage makes sense
- Leadership efficiency visible
- Failed claims less costly
- No resource depletion

### 3. Time Mechanics
- Time accumulation logical
- Leadership speed bonus works
- Stage timing reasonable
- Overall completion possible

### 4. Information Mechanics
- Information flows properly
- Quality improves logically
- Leadership emerges from sharing
- Stage completion works

## Next Steps If Successful
1. Test with different parameters
2. Add contexts
3. Compare perspectives

## Next Steps If Issues
1. Debug stage progression
2. Adjust resource mechanics
3. Fix timing issues
4. Improve information flow 