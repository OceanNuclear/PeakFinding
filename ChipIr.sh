#!/bin/sh
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/01_Cu_*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/01_Cu
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/02_Cu_00?_elevated.Spe ~/Documents/PhD/ExperimentalVerification/analysis/02_Cu_elevated
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/02_Cu_00?.Spe ~/Documents/PhD/ExperimentalVerification/analysis/02_Cu_contact
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/02_Cu_00?_elevated.Spe ~/Documents/PhD/ExperimentalVerification/data/experiment/02_Cu_00?.Spe ~/Documents/PhD/ExperimentalVerification/analysis/02_Cu_all
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/03_In_*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/03_In
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/04_Sn*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/04_Sn
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/05_Ge0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/05_Ge
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/06_Zn0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/06_Zn
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/07_Pt0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/07_Pt
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/08_Hf0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/08_Hf
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/09_Bag0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/09_Bag
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/10_Yt0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/10_Yt
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/11_Mo0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/11_Mo
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/12_Cd0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/12_Cd
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/13_Er0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/13_Er
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/14_Nd0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/14_Nd
python $1 ~/Documents/PhD/ExperimentalVerification/data/experiment/15_Bi0*.Spe ~/Documents/PhD/ExperimentalVerification/analysis/15_Bi