README
================


## Reproducibility
Randomization seed is set at the beginning of all R scripts, to ensure rproducibility of results.\

To reproduce results, graphs & tables in the report, do the following R scrips sequentially:\

(1): Ensure there is a data folder in directory, with imagem1.txt, imagem2.txt, imagem3.txt in it\

(2): Run 'EDA.R': This generates EDA figures & Tables \

(3): Run 'Run_CV.R': This runs 5-fold Cross-validation on all the models, including parameter tuning. It will take some time. CV results are saved in .rds
file inside data folder\

(4): Run 'Performance.R': This loads CV results from 'Run_CV.R' which is saved in data folder, and generagte CV results & Test performance tables & figures.\

(5): Run 'Diagnostic.R': This generates all the figures in Diagnostic sesction.\

(6): Run 'Diagnostic2.R': This generates all the figures in Diagnostic section using K-means data split.\

'CVmaster.R' contains the generic function to perform the Cross-Validation. It also contains 'block_split' & 'Kmeans_block_split' functions which
creates the data splitting.\
