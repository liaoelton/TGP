(If you want to test realworld datasets, PMLB datasets needs to be installed first, refer to https://epistasislab.github.io/pmlb/using-python.html or you can use following codes)
"""
git clone https://github.com/EpistasisLab/pmlb
cd pmlb/
pip install .
""" 

# generate randon tree for NN model 
cd gp_tree_data
python gptree_random_generator.py
cp GP_rand_tree.json ../LSTM_model/train.json
cd ..

# trainig LSTM model
cd LSTM_model/
python train_LSTM.py
cp ckpt/output/best-model.pth ../TGP/gplearn
cd ..

# run GP
cd TGP
python runGP.py -t <exp_runs> -g <generation> -p <population> -d <data_name> -n <file_name>  -s <is_SGP>

* data_name(string)
We have the following options: 
"example_1", "example_2" , "example_3" for our toy problems
example_1 : x^1 + x^2 + x^3 + x^4 + x^5
example_2 : x^1 - x^2  + x^3 + x^4 + x^5
example_3 : x^2 - x^4 + x^4


"real_world" for PMLB real world datasets


* is_SGP(bool)
true :  using simple genetic programming(SGP) 
false : using our model (LSTM+GP)

