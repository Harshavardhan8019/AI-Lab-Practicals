# AI Lab Practicals 🤖

A comprehensive collection of Artificial Intelligence laboratory practicals covering foundational AI techniques: Constraint Satisfaction Problems (CSP), Probabilistic Reasoning, Natural Language Processing & Text Classification, Automated Planning (STRIPS), Reinforcement Learning (Markov Decision Processes & Value Iteration), Propositional Logic Inference, and Supervised Machine Learning (k-NN).

---

## 📚 Table of Contents

- [Overview](#overview)
- [Practicals Index](#practicals-index)
  - [Practical 6: Map Coloring Using Backtracking (CSP)](#practical-6-map-coloring-using-backtracking-csp)
  - [Practical 7: Simple Probabilistic Classifier for Diagnosis](#practical-7-simple-probabilistic-classifier-for-diagnosis)
  - [Practical 8: Mini AI Text Classification Application](#practical-8-mini-ai-text-classification-application)
  - [Practical 9: STRIPS / Goal-Stack Planning](#practical-9-strips--goal-stack-planning)
  - [Practical 10: Value Iteration in a Simple Grid World](#practical-10-value-iteration-in-a-simple-grid-world)
  - [Practical 11: Propositional Logic and Logical Inference](#practical-11-propositional-logic-and-logical-inference)
  - [Practical 12: Simple AI Learning Model Using k-NN](#practical-12-simple-ai-learning-model-using-k-nn)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Technologies & Libraries](#technologies--libraries)

---

## 🔍 Overview

This repository contains hands-on Jupyter Notebook implementations of core AI algorithms and concepts:

| # | Practical Title | Domain / Technique | Key Concept |
|---|-----------------|-------------------|-------------|
| **06** | Map Coloring Using Backtracking | Constraint Satisfaction (CSP) | Backtracking, Graph Coloring, Adjacency Constraints |
| **07** | Simple Probabilistic Classifier | Probabilistic Reasoning / Bayes | Prior, Likelihood, Posterior, Medical Diagnosis |
| **08** | Mini AI Text Classification App | NLP & Supervised Classification | Bag of Words (`CountVectorizer`), Naive Bayes, Explainability & Ethics |
| **09** | STRIPS / Goal-Stack Planning | Classical AI Planning | Preconditions, Add/Delete Effects, Goal-Stack Decomposition |
| **10** | Value Iteration in Grid World | Reinforcement Learning / MDP | Bellman Equation, Discount Factor ($\gamma$), Optimal Policy Extraction |
| **11** | Propositional Logic & Inference | Knowledge Representation | Truth Tables, Logical Inference, Modus Ponens |
| **12** | Supervised Learning with k-NN | Machine Learning | Euclidean Distance, Majority Voting, Decision Boundaries |

---

## 📖 Practicals Detailed Breakdown

### Practical 6: Map Coloring Using Backtracking (CSP)
* **File:** [`Practical_06_Map_Coloring_CSP.ipynb`](./Practical_06_Map_Coloring_CSP.ipynb)
* **Objective:** Solve a constraint satisfaction problem (CSP) using the backtracking search algorithm.
* **Problem:** Assign colors to regions/nodes on a map such that no two adjacent regions share the same color.
* **Key Steps:**
  1. Define regions and their adjacency graph.
  2. Implement constraint checking function (`is_valid`).
  3. Recursively assign colors using backtracking when conflicts occur.

---

### Practical 7: Simple Probabilistic Classifier for Diagnosis
* **File:** [`Practical_07_Probabilistic_Classifier.ipynb`](./Practical_07_Probabilistic_Classifier.ipynb)
* **Objective:** Model a simple probabilistic reasoning system to diagnose illness given symptoms.
* **Problem:** Given observed symptoms (e.g., Fever, Cough), compute the posterior probability of candidate diagnoses using Bayes' Rule:
  $$P(\text{Diagnosis} \mid \text{Symptoms}) \propto P(\text{Symptoms} \mid \text{Diagnosis}) \cdot P(\text{Diagnosis})$$
* **Key Steps:**
  1. Define prior disease probabilities and conditional symptom likelihoods.
  2. Calculate unnormalized posterior probabilities for patient symptom sets.
  3. Normalize probabilities to determine the most likely diagnosis.

---

### Practical 8: Mini AI Text Classification Application
* **File:** [`Practical_08_Text_Classification_Spam_Detection.ipynb`](./Practical_08_Text_Classification_Spam_Detection.ipynb)
* **Objective:** Build an end-to-end text classification pipeline for Spam vs. Ham detection, and examine AI ethics and explainability.
* **Key Steps:**
  1. Prepare text dataset labeled as `Spam` and `Not Spam`.
  2. Convert text to numerical features using `CountVectorizer`.
  3. Train a `MultinomialNB` (Multinomial Naive Bayes) classifier.
  4. Test live predictions on new user input.
  5. Discuss transparency, explainability, false positives, and privacy considerations.

---

### Practical 9: STRIPS / Goal-Stack Planning
* **File:** [`Practical_09_STRIPS_Goal_Stack_Planning.ipynb`](./Practical_09_STRIPS_Goal_Stack_Planning.ipynb)
* **Objective:** Implement automated planning for a robot navigation problem using the STRIPS representation and Goal-Stack planning.
* **Key Concepts:**
  - **Preconditions:** Facts that must hold before executing an action.
  - **Add List:** Facts that become true after the action.
  - **Delete List:** Facts that cease to be true after the action.
  - **Goal-Stack:** Working backwards from the goal state to satisfy sub-goals recursively.

---

### Practical 10: Value Iteration in a Simple Grid World
* **File:** [`Practical_10_Value_Iteration_Grid_World.ipynb`](./Practical_10_Value_Iteration_Grid_World.ipynb)
* **Objective:** Implement Value Iteration for sequential decision-making in a 2×2 Grid World environment with terminal states.
* **Key Concepts:**
  - States, Transitions, Rewards, and Discount Factor ($\gamma = 0.9$).
  - Bellman Value Update Equation:
    $$V(s) \leftarrow \max_a \left( R(s') + \gamma V(s') \right)$$
  - Policy extraction to find the optimal path to the goal state $G$.

---

### Practical 11: Propositional Logic and Logical Inference
* **File:** [`Practical_11_Propositional_Logic_Inference.ipynb`](./Practical_11_Propositional_Logic_Inference.ipynb)
* **Objective:** Represent logical rules and real-world scenarios using Propositional Logic and perform logical inference.
* **Key Concepts:**
  - Propositions, truth values (`True` / `False`), and logical operators (`AND`, `OR`, `NOT`, `IMPLIES`).
  - Knowledge Base (KB) representation.
  - Truth table enumeration and inference rules (Modus Ponens) to derive guaranteed conclusions.

---

### Practical 12: Simple AI Learning Model Using k-NN
* **File:** [`Practical_12_KNN_Learning_Model.ipynb`](./Practical_12_KNN_Learning_Model.ipynb)
* **Objective:** Implement and evaluate the $k$-Nearest Neighbors ($k$-NN) algorithm for pattern recognition and classification.
* **Key Concepts:**
  - Feature spaces and distance metrics (Euclidean distance).
  - Choosing hyperparameter $k$.
  - Majority voting among neighbors to classify query instances.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Harshavardhan8019/AI-Lab-Practicals.git
   cd AI-Lab-Practicals
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install jupyter notebook scikit-learn numpy matplotlib
   ```

---

## 🚀 How to Run

Launch Jupyter Notebook in the repository directory:
```bash
jupyter notebook
```
or run JupyterLab:
```bash
jupyter lab
```
Click on any `.ipynb` file to view the explanations and run the code cells sequentially.

---

## 💻 Technologies & Libraries

- **Language:** Python 3.x
- **Environment:** Jupyter Notebook / Google Colab
- **Libraries:**
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning models & text vectorization
  - `matplotlib` - Data visualization (where applicable)

---

## 👤 Author

* **Harshavardhan Thota** - [@Harshavardhan8019](https://github.com/Harshavardhan8019)
