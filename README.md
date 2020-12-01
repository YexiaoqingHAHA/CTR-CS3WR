# CTR-CS3WR
This is the Github repository containing the code for "An interpretable sequential three-way recommendation based on collaborative topic regression"

## The whole program including downloading the datasets and training the model,  making the recommendation and processing the results
## data
http://www.citeulike.org/faq/data.adp
## train the model
sh run_a.sh 
or 
sh run_t.sh
##  make the recommendation
cd ./CS3WR
python main.py
##  processing the results
cd ./CS3WR
python process_results.py
