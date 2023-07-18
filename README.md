# Energy Balance Model

The code in this repository is used to study the sensitivity of the nocturnal and polar atmospheric boundary layer to transient phenomena. 
The analysis is performed with a  model for near-surface temperature inversion transitions which was originally defined by van de Wiel et. al. [1] but has been modified
to include different types of perturbations. 

The corresponding publication is:
 Kaiser, A., Vercauteren, N., and Krumscheid, S.: Sensitivity of the nocturnal and polar boundary layer to transient phenomena, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-1519, 2023.

If comments in the code include references to figures or equations they refer to the ones in this publication.

## How to set up and run the model
1. Clone repository
```bash
git clone git@github.com:am-kaiser/energy_balance_model.git
cd energy_balance_model
```

2. Create and activate environment
using a virtual environment:
```bash
module load Python/3.9.5-GCCcore-10.3.0 # depending on system set up
pip install virtualenv # if you don't have virtualenv installed
virtualenv ebm_venv
source ebm_venv/bin/activate
pip install -r docker/requirements.txt
```
alternatively using docker:
```bash
cd docker
docker build . -t energy_balance_model 
cd ..
docker run -ti -v "$PWD:/home/" energy_balance_model
cd home
```

3. Run the model
```bash
python main.py # when using virtualenv
python3 main.py # docker
```
Output files can be found in output/ directory with current data and time.

### Change model parameters
All model parameters can be set in the [file](one_D_model/model/parameters.py). For a description of most of the model parameters see section 2.1.

### Command-line options for running the model

```bash
usage: main.py [-h] [-pl] [-V] [-odeu] [-f] [-u] [-uf] [-sf] [-sfu] [-ss] [-a]
                                                                              
Run the energy balance model.                                                 
                                                                              
optional arguments:                                                           
  -h, --help            show this help message and exit                       
  -pl, --plot           Make potential, bifurcation, ... plots.               
  -V, --version         show program's version number and exit                
  -odeu, --ode_with_var_u                                                     
                        Run ODE model with time dependent u.                  
  -f, --function        Randomize the model function itself.                  
  -u, --u               Randomize the model parameter u.
  -uf, --u_and_function
                        Randomize the model parameter u and the model itself.
  -sf, --stab_function  Randomize the stability function.
  -sfu, --stab_function_and_time_dependent_u
                        Randomize the stability function and u is time dependent.
  -ss, --sensitivity_study
                        Perform a sensitivity study.
  -a, --all             Run model with all randomizations.
```
## Contributors
* [Amandine Kaiser](https://github.com/am-kaiser)
* [Nikki Vercauteren](https://github.com/vercaute) (support)

## Disclaimer
This code has been set up to run on Ubuntu 20.04. 
Only the python versions 3.8 and 3.9 have been used to run the code.

There is no guarantee that the code runs on any other operating system or with a different python version.

## References
[1] van de Wiel, B. J. H., Vignon, E., Baas, P., van Hooijdonk, I. G. S., van der Linden, S. J. A., Antoon van Hooft, J., Bosveld, F. C., de Roode,
S. R., Moene, A. F., and Genthon, C.: Regime Transitions in Near-Surface Temperature Inversions: A Conceptual Model, Journal of
Atmospheric Sciences, 74, 1057–1073, https://doi.org/10.1175/JAS-D-16-0180.1, 2017.