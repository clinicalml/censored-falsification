# censored-falsification
Code for paper, `Testing the validity of observational studies with right censoring`


`conda create env -n falsification python=3.9`

`poetry install`


You can install poetry here :

`curl -sSL https://install.python-poetry.org | python3 -`

Then you can run it like that (you can specify the censoring cases there):

`python run_mmr.py censoring=conditional_censoring`