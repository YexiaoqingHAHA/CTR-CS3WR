# CTR-CS3WR
This repository contains the code of the paper  "An interpretable sequential three-way recommendation based on collaborative topic regression"

It benefited greatly from https://github.com/blei-lab/ctr

The whole program including four parts: downloading the datasets, training the model,  making the recommendation and processing the results.

## 1.data
http://www.citeulike.org/faq/data.adp

## 2.model training
  `sh run_a.sh`

or 

  sh run_t.sh
## 3.recommendation making
  cd ./CS3WR

  python main.py
## 4.results processing
  cd ./CS3WR

  python process_results.py
