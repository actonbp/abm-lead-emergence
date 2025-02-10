# Base Leadership Task - Pseudo Code Documentation

## Core Concept
The base task implements leadership emergence through information sharing, where:
- Leadership claims are made by sharing information
- Leadership grants are accepting shared information
- Group success depends on efficient information integration
- Established leaders gain efficiency benefits

## Task Structure

```python
class BaseLeadershipTask:
    """
    A task where agents must collectively discover truth through information sharing.
    Leadership emerges through the process of sharing and accepting information.
    """
    
    def __init__(self, num_agents: int):
        # Initialize task parameters
        self.num_agents = num_agents
        self.initial_resources = 100.0
        self.current_resources = self.initial_resources
        self.time_spent = 0
        
        # Generate true solution (what group needs to discover)
        self.true_solution = generate_random_pattern(size=10)
        
        # Track group's current understanding
        self.group_understanding = initialize_neutral()
        
        # Track what information has been shared
        self.shared_info = set()
        
        # Track leadership status
        self.current_leader = None
        self.leadership_history = []
        
        # Distribute information to agents
        self.agent_info = distribute_information()
    
    def distribute_information():
        """
        Give each agent a mix of accurate and misleading information.
        Some information overlap ensures multiple paths to solution.
        """
        agent_info = {}
        for agent_id in range(num_agents):
            pieces = []
            for _ in range(pieces_per_agent):
                if random() < 0.7:  # 70% accurate
                    value = true_value + small_noise()
                else:  # 30% misleading
                    value = opposite_of(true_value) + small_noise()
                pieces.append(value)
            agent_info[agent_id] = pieces
        return agent_info
    
    def make_leadership_claim(claiming_agent, receiving_agent):
        """
        Agent attempts to lead by sharing information.
        Returns success/failure and associated costs.
        """
        # Get unshared information from claiming agent
        available_info = get_unshared_info(claiming_agent)
        if not available_info:
            return {"success": False, "reason": "no_info"}
            
        # Select piece to share
        info_piece = select_random(available_info)
        
        # Calculate base costs
        base_resource_cost = 5.0
        base_time_cost = 1.0
        
        # Apply leadership efficiency bonus if applicable
        if claiming_agent == current_leader:
            base_resource_cost *= 0.5
            base_time_cost *= 0.5
        
        return {
            "info": info_piece,
            "resource_cost": base_resource_cost,
            "time_cost": base_time_cost
        }
    
    def process_leadership_grant(claim_info, granted: bool):
        """
        Process the outcome of a leadership claim.
        Updates group understanding and resources.
        """
        if granted:
            # Store old state for comparison
            old_understanding = group_understanding.copy()
            
            # Integrate new information
            update_group_understanding(claim_info["info"])
            
            # Calculate improvement
            improvement = evaluate_improvement(
                old_understanding, 
                group_understanding
            )
            
            # Update resources
            spend_resources(claim_info["resource_cost"])
            add_time(claim_info["time_cost"])
            
            # Update leadership if significant improvement
            if improvement > LEADERSHIP_THRESHOLD:
                update_leadership(claiming_agent)
            
            return {
                "success": True,
                "improvement": improvement,
                "costs": {
                    "resources": claim_info["resource_cost"],
                    "time": claim_info["time_cost"]
                }
            }
        else:
            # Failed claims still cost (but less)
            spend_resources(claim_info["resource_cost"] * 0.3)
            add_time(claim_info["time_cost"] * 0.3)
            
            return {
                "success": False,
                "improvement": 0,
                "costs": {
                    "resources": claim_info["resource_cost"] * 0.3,
                    "time": claim_info["time_cost"] * 0.3
                }
            }
    
    def evaluate_solution():
        """
        Evaluate current group performance.
        Combines solution quality with resource efficiency.
        """
        # Calculate solution quality (60% weight)
        quality = calculate_similarity(
            group_understanding,
            true_solution
        )
        
        # Calculate efficiency (40% weight)
        resource_efficiency = current_resources / initial_resources
        time_efficiency = 1.0 - (time_spent / max_time)
        efficiency = (resource_efficiency + time_efficiency) / 2
        
        # Combine scores
        return 0.6 * quality + 0.4 * efficiency
```

## Key Components

### 1. Information Structure
- True solution: Pattern group needs to discover
- Agent information: Mix of accurate (70%) and misleading (30%) pieces
- Group understanding: Current collective knowledge

### 2. Leadership Mechanics
- Claims made by sharing information
- Grants = accepting shared information
- Leadership status tracked and provides efficiency bonus
- Failed claims have reduced but non-zero costs

### 3. Resource Management
- Fixed initial resource pool
- Costs for sharing information
- Leadership reduces costs by 50%
- Time tracking for efficiency

### 4. Performance Evaluation
- Solution quality (similarity to truth)
- Resource efficiency
- Time efficiency
- Weighted combination for final score

## Theoretical Alignment

1. **Leadership Through Information**
   - Leadership claims = information sharing
   - Leadership grants = information acceptance
   - Success measured by moving group toward truth

2. **Leadership Benefits**
   - Established leaders more efficient
   - Failed claims still costly
   - Group benefits from stable leadership

3. **Group Performance**
   - Quality of solution
   - Efficiency of process
   - Balance between speed and accuracy

4. **Emergence Mechanisms**
   - Natural selection of effective leaders
   - Cost-benefit trade-offs
   - Group learning through interaction

## Usage in Different Contexts

### 1. High Uncertainty
- Increase noise in information
- Add dynamic changes to true solution
- Reduce initial information accuracy

### 2. Resource Constraints
- Lower initial resources
- Increase base costs
- Larger leadership efficiency bonus

### 3. Time Pressure
- Add strict time limit
- Increase time costs
- Larger time efficiency weight

## Implementation Notes

1. **Simplicity**
   - Keep core mechanics simple
   - Add complexity through contexts
   - Clear connection to theory

2. **Flexibility**
   - Easy to modify parameters
   - Adaptable to different contexts
   - Measurable outcomes

3. **Theoretical Value**
   - Clear leadership emergence
   - Meaningful group performance
   - Testable hypotheses 