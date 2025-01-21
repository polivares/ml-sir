# Epidemic Spreading and Machine Learning

This repository contains the code and explorations conducted as part of the project combining **epidemiological models** and **machine learning techniques** to analyze and predict the spread of infectious diseases. The goal is to generate insights and tools to improve parameter estimation and forecasting capabilities for epidemiological phenomena.

## Project Overview
The project is structured around the integration of the **SIR epidemiological model** and **machine learning techniques**. Key objectives include:

- **Simulating Data:** Using the SIR model to generate synthetic datasets with tunable parameters.
- **Parameter Estimation:** Exploring traditional optimization methods and neural networks to estimate key parameters (e.g., \(\beta\) and \(\gamma\)).
- **Performance Comparison:** Comparing the accuracy and efficiency of different approaches.

## Repository Structure

```
├── data/                     # Data directory
│   ├── external/             # External datasets
│   ├── interim/              # Intermediate processed data
│   ├── processed/            # Final processed datasets
│   └── raw/                  # Raw datasets
│       └── simulated/        # Simulated SIR datasets
├── docs/                     # Project documentation
├── models/                   # Trained models and results
├── notebooks/                # Jupyter Notebooks for exploration
│   ├── exploratory/          # Main notebooks for data exploration
│   │   ├── markdown/         # Notebooks exported as Markdown
│   │   ├── models/           # Model-specific notebooks
│   │   └── old_exploratory/  # Older exploratory notebooks
├── references/               # Relevant references and research papers
├── reports/                  # Generated reports and visualizations
├── src/                      # Source code for the project
│   ├── data/                 # Scripts for data processing
│   ├── features/             # Scripts for feature engineering
│   ├── models/               # Scripts for model training and evaluation
│   └── visualization/        # Scripts for creating visualizations
├── .gitignore                # Files and directories to ignore in git
├── LICENSE                   # License information
├── Makefile                  # Commands for project automation
├── README.md                 # Project documentation (this file)
├── requirements.txt          # Python dependencies
├── setup.py                  # Installation script
├── test_environment.py       # Environment testing script
└── tox.ini                   # Configuration for Tox testing
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/polivares/ml-sir.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ml-sir
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Notebooks

### 1. **SIR Data Generation**
   - File: `notebooks/exploratory/old_exploratory/0.1-por-SIR_data_generation.ipynb`
   - Generates synthetic datasets using the SIR model.
   - Outputs time series data for susceptible (S), infected (I), and recovered (R) compartments, along with their corresponding parameters \(\beta\) and \(\gamma\).

### 2. **Parameter Fitting (Traditional Methods)**
   - File: `notebooks/exploratory/old_exploratory/1.1-por-SIR_data_fit_traditional.ipynb`
   - Uses optimization techniques (e.g., least squares) to estimate parameters \(\beta\) and \(\gamma\) from the generated data.
   - Compares estimated parameters with ground truth values.

### 3. **Parameter Fitting (Neural Networks)**
   - File: `notebooks/exploratory/old_exploratory/2.1-por-SIR_data_fit_NN.ipynb`
   - Implements a neural network to predict \(\beta\) and \(\gamma\) directly from the infected compartment data.
   - Achieved high performance with an R2 score of ~0.9943 on test data.

## Results
Soon...
<!--### Traditional Methods
- Parameter estimation using optimization techniques provided accurate results but required iterative processes for each dataset.

### Neural Networks
- Achieved fast and accurate predictions for \(\beta\) and \(\gamma\), demonstrating the potential of machine learning for parameter estimation in epidemiology.-->

<!--## Contributing
Contributions to improve the project are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.
-->

## Contact
For questions or suggestions, please contact:
- **Project Lead:** Patricio Olivares R.
- **Email:** patricio.olivaresr@usm.cl

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
