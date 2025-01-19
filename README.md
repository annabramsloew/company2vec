# Company2Vec: Modeling Company Trajectories through Life-event Sequences
**Master's Thesis @ DTU Compute by Nikolai Beck Jensen & Anna Bramsløw**

For our Master's Thesis, we extend the principles of life2vec, originally developed for modeling human life trajectories, to analyze and predict company lives. By representing companies as structured sequences of life events, we develop a symbolic language capable of capturing the complexity of corporate trajectories. Utilizing over 29 million event registrations from more than 400,000 Danish firms spanning 2013–2023, we present company2vec, a model that learns semantically meaningful representations of these sequences using state-of-the-art methods from language modeling. 

We have 4 prediction tasks:
- Bankruptcy
- Capital Increase
- Firm Relocation (moving in code)
- Development in Employee levels (employees in code)

This repository contains all code produced for the project, including exploratory analyses and analysis of results. The modeling architecture has been developed based on Germans Savcisens Ph.D. dissertation and corresponding open-source code.

We refer to Savcisens' repository for guides on how to run experiments, etc.

The structure of our repository is as follows:
- **notebooks**: analyses of data and results
- **src**: model architecture, symbolic language processing, result metrics and utils
- **conf**: config files to run experiments
- **data_processing**: data fetching module connecting to the virk API

Sources:
- life2vec publication: https://www.nature.com/articles/s43588-023-00573-5
- life2vec Github Repository: https://github.com/SocialComplexityLab/life2vec/tree/main
