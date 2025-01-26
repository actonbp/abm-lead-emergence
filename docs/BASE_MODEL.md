# Base Leadership Emergence Model Structure

## Core Components

| Component | Description | Key Features |
|-----------|-------------|--------------|
| ModelParameters | Parameter validation schema | - Core parameters (n_agents, identities, change rates)<br>- ILT matching parameters<br>- Interaction parameters<br>- Network parameters |
| Agent | Individual agent in simulation | - Identity states (leader/follower)<br>- Characteristics and ILT<br>- Interaction history<br>- Decision-making methods |
| BaseLeadershipModel | Main simulation engine | - Agent management<br>- Interaction processing<br>- Network tracking<br>- History recording |

## Model Parameters

| Parameter Category | Parameter | Type | Default | Description |
|-------------------|-----------|------|---------|-------------|
| Core Parameters | n_agents | int | - | Number of agents (2-100) |
| | initial_li_equal | bool | True | Whether agents start with equal leadership identities |
| | initial_identity | float | 50.0 | Initial identity value if equal (0-100) |
| | li_change_rate | float | 2.0 | Rate of leadership identity change (0-5) |
| ILT Matching | ilt_match_algorithm | str | "euclidean" | Algorithm for ILT matching |
| | ilt_match_params | dict | See below | Parameters for matching algorithm |
| Interaction | claim_multiplier | float | 0.7 | Multiplier for claim probability |
| | grant_multiplier | float | 0.6 | Multiplier for grant probability |
| Network | interaction_radius | float | 1.0 | Radius for agent interactions |
| | memory_length | int | 0 | Number of past interactions to remember |

### ILT Match Parameters

| Algorithm | Parameters | Default Values |
|-----------|------------|----------------|
| Gaussian | sigma | 20.0 |
| Sigmoid | k | 10.0 |
| Threshold | threshold | 15.0 |

## Agent State

| Attribute | Type | Description |
|-----------|------|-------------|
| id | int | Unique identifier |
| leader_identity | float | Current leader identity (0-100) |
| follower_identity | float | Current follower identity (0-100) |
| characteristics | float | Leadership characteristics (40-60) |
| ilt | float | Implicit Leadership Theory value (40-60) |
| history | dict | Tracks identity changes over time |
| last_interaction | dict | Details of most recent interaction |

## Model Methods

### Core Simulation Methods

| Method | Purpose | Key Operations |
|--------|---------|----------------|
| step() | Execute one simulation step | - Select interaction pair<br>- Process interaction<br>- Update states<br>- Track outcomes |
| run() | Run multiple simulation steps | - Execute steps<br>- Collect states<br>- Return history |

### Internal Processing Methods

| Method | Purpose | Operations |
|--------|---------|------------|
| _select_interaction_pair() | Choose agents for interaction | Random selection of two agents |
| _process_interaction() | Handle agent interaction | - Evaluate claims/grants<br>- Calculate probabilities |
| _update_identities() | Update agent states | Modify identities based on interaction |
| _update_network() | Update interaction network | Add/update network edges |
| _track_outcomes() | Record simulation state | Store metrics and states |

## History Tracking

| Metric | Type | Description |
|--------|------|-------------|
| leader_identities | list[float] | Leader identity values over time |
| follower_identities | list[float] | Follower identity values over time |
| centralization | list[float] | Network centralization measures |
| density | list[float] | Network density measures |
| interaction_patterns | list[dict] | Network structure over time |

## Model State Output

| Field | Type | Description |
|-------|------|-------------|
| time | int | Current simulation step |
| agents | list[dict] | Current state of all agents |
| network | NetworkX Graph | Current interaction network |
| leader_identities | list[float] | Current leader identity values |
| follower_identities | list[float] | Current follower identity values |
| centralization | float | Current network centralization |
| density | float | Current network density 