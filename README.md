# Exact and Affordable Option Probabilities for Large Business Label Sets

Code, inputs and results for the paper *Exact and Affordable Option Probabilities for Large Business Label Sets: Cached Scoring of LLM Decisions for Intent Routing*.

A language model can route requests over a large intent taxonomy only if it gives every intent a probability that thresholds can act on. Reading the probability of every numbered option exactly, node by node over the prefix tree of option numbers, repeats the prompt once per tree node. The study compares this prefix-tree readout with a cached scorer that runs the prompt once and scores all option numbers on copies of its key-value cache, and with three shortcuts (first answer token, letters A-Z, generation with a stated confidence), on nine label sets of 3 to 151 intents (152,000 decisions).

## Article
* **Journal**: Under Review
* **Title**: Exact and Affordable Option Probabilities for Large Business Label Sets: Cached Scoring of LLM Decisions for Intent Routing
* **DOI**: 

## Authors
* **Adj. Asst. Prof. Konstantinos I. Roumeliotis | University of Peloponnese**
* **Asoc. Prof. Dionisis Margaris | University of Peloponnese**
* **Asoc. Prof. Dimitris Spiliotopoulos | University of Peloponnese**
* **Prof. Nikolaos D. Tselikas | University of Peloponnese**