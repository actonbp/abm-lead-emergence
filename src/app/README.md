# Interactive Application

This directory contains the interactive visualization and analysis application for the leadership emergence project. The app provides a user-friendly interface for running simulations and analyzing results.

## Components

- `app.py`: Main Shiny application setup and UI definition
  - Defines the application layout
  - Sets up reactive elements
  - Handles user interactions

- `app_analysis.py`: Analysis components for the app
  - Processes simulation results
  - Generates plots and visualizations
  - Computes summary statistics

- `app_step.py`: Step-by-step simulation visualization
  - Handles simulation stepping
  - Updates network visualization
  - Shows agent state changes

- `streamlit_app.py`: Alternative Streamlit implementation
  - Provides similar functionality to Shiny app
  - Uses Streamlit for UI components
  - Currently in development

## Integration

The app integrates with the core simulation and analysis code by:
1. Importing model classes from `src/models/`
2. Using analysis functions from `src/analysis/`
3. Leveraging visualization tools from `src/visualization/`

## Development Status

This is a work in progress. Current priorities:
1. Completing the basic simulation visualization
2. Adding parameter exploration capabilities
3. Implementing real-time analysis features

## Running the App

For the Shiny app:
```R
library(shiny)
runApp("src/app/app.py")
```

For the Streamlit app:
```bash
streamlit run src/app/streamlit_app.py
``` 