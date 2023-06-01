# Energy Balance Model

The code in this repository is used to study the sensitivity of the nocturnal and polar atmospheric boundary layer to transient phenomena. 
The analysis is performed with a  model for near-surface temperature inversion transitions which was originally defined by van de Wiel et. al. [1] but has been modified
to include perturbations. 

## How to set up and run the model
1. Clone repository
```bash
git clone git@github.com:am-kaiser/energy_balance_model.git
cd energy_balance_model
```

2. Create and activate environment
```bash
module load Python/3.9.5-GCCcore-10.3.0 # depending on system set up
pip install virtualenv # if you don't have virtualenv installed
virtualenv ebm_venv
source ebm_venv/bin/activate
pip install -r docker/requirements.txt
```
alternatively a Docker image can be build with
```bash
docker build . -t energy_balance_model 
```

3. Run the model
```bash
python main.py
```
Output files can be found in output/ directory with current data and time.

### Command-line options for running the model

```bash
usage: main.py [-h] [-pl] [-V] [-f] [-sf] [-sfmn] [-qi] [-l] [-u] [-uf] [-sfu] [-a]
                                                                                   
Run the energy balance model.                                                      
                                                                                   
optional arguments:                                                                
  -h, --help            show this help message and exit                            
  -pl, --plot           Make potential, bifurcation, ... plots.                    
  -V, --version         show program's version number and exit                     
  -f, --function        Randomize the model function itself.                       
  -sf, --stab_function  Randomize the stability function.                          
  -sfmn, --stab_function_multi_noise
                        Randomize the stability function.
  -qi, --Qi             Randomize the model parameter Qi.
  -l, --Lambda          Randomize the model parameter lambda.
  -u, --u               Randomize the model parameter u.
  -uf, --u_and_function
                        Randomize the model parameter u and the model itself.
  -sfu, --stab_function_multi_noise_u_td
                        Randomize the stability function and u is timedependent.
  -a, --all             Run model with all randomizations.

```
## Disclaimer
This code has been set up to run on Ubuntu 20.04. 
Only the python versions 3.8 and 3.9 have been used to run the code.

There is no guarantee that the code runs on any other operating system or with a different python version.

## References
[1] van de Wiel, B. J. H., Vignon, E., Baas, P., van Hooijdonk, I. G. S., van der Linden, S. J. A., Antoon van Hooft, J., Bosveld, F. C., de Roode,
S. R., Moene, A. F., and Genthon, C.: Regime Transitions in Near-Surface Temperature Inversions: A Conceptual Model, Journal of
Atmospheric Sciences, 74, 1057–1073, https://doi.org/10.1175/JAS-D-16-0180.1, 2017.