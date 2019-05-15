- Criar um ambiente no anaconda: 
conda create --name nome_ambiente

- Entrar no ambiente: 
source activate nome_ambiente

- Instalar bibliotecas: 

conda install -c anaconda tensorflow 
conda install -c conda-forge keras 
conda install -c anaconda scikit-learn 
conda install -c conda-forge matplotlib
conda install -c anaconda pydot

Executar o arquivo para realizar o treinamento e gerar o modelo:
python3 lenet5_mnist.py >> output.log


