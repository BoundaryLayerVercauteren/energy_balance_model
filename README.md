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
pip install virtualenv # if you don't have virtualenv installed
virtualenv ebm_venv
source ebm_venv/bin/activate
pip install -r requirements.txt
```

3. Run the model
```bash
python main.py
```
Output files can be found in output/ directory with current data and time.

### Command-line options for running the model

```bash
usage: main.py [-h] [-pl] [-V] [-f] [-sf] [-qi] [-l] [-z] [-u] [-a] [-obs_u]

Run the energy balance model.

optional arguments:
  -h, --help            show this help message and exit
  -pl, --plot           Make potential, bifurcation, ... plots.
  -V, --version         Show program's version number and exit
  -f, --function        Randomize the model function itself.
  -sf, --stab_function  Randomize the stability function.
  -qi, --Qi             Randomize the model parameter Qi.
  -l, --Lambda          Randomize the model parameter lambda.
  -z, --z0              Randomize the model parameter z0.
  -u, --u               Randomize the model parameter u.
  -a, --all             Run model with all randomizations.
  -obs_u, --observed_u  Run model with wind speed given by observations.
```
## Disclaimer
This code has been set up to run on Ubuntu 20.04. There is no guarantee that it runs on any other operating system.

## References
[1] van de Wiel, B. J. H., Vignon, E., Baas, P., van Hooijdonk, I. G. S., van der Linden, S. J. A., Antoon van Hooft, J., Bosveld, F. C., de Roode,
S. R., Moene, A. F., and Genthon, C.: Regime Transitions in Near-Surface Temperature Inversions: A Conceptual Model, Journal of
Atmospheric Sciences, 74, 1057–1073, https://doi.org/10.1175/JAS-D-16-0180.1, 2017.