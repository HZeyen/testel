# testel
 # RefraPy manual

Program reads a number of seg2 or segy input files and allows certain data treatment like different filters and picking with different methods. Different plotting options are given like gaining, shot/ receiver/ distance gathers.

## Installation
1. Mouse 

    It is strongly recommended to use a mouse. The program needs the central mouse button (or wheel). If you are working with a touchpad, configure it such that it simulates the central mouse button. The simplest is certainly to configure a three-finger-touch as central button. For this, under Windows press the WINDOWS-Key+I. On the screen that appears, click on “Périfériques”. There, on the left-side panel, search “Pavé tactile”. Scroll down until you see the image with three fingers. Below this image, under “Appuis”, change to “Button central de la souris”.

2. Copy the following files all into the same folder of your choice: 

+ refraPy.py (the main program)
+ refraData.py
+ refraPlot.py
+ refraWindow.ui
3. Download and install Anaconda Individual Edition
4. Push the Windows key and search Anaconda3. There, choose (**RIGHT CLICK ON IT**) Anaconda Prompt and execute (if possible) as administrator. 
This opens a command window. Then type the following commands:
    + `conda update --all`
    + `conda config --add channels gimli --add channels conda-forge`
« channels » is the place where to find the source code, here « gimli » and “conda-forge”
    + `conda create -n pg pygimli pybert`
(this may take quite some time, don’t worry about error messages as long as conda is running). Here, “pg” means the new environment called “pg”. Pybert is not necessary for refraPy, but if you want also to use Orsay ERT inversion programs, better to install everything together.
        + `conda activate pg` (activation of the environment “pg”)
        + `conda install obspy`
        + `conda install statsmodels`
        + `conda update –all`  (just to be sure to have the latest versions)
        
5. Open Anaconda Navigator (Windows key -> Anadconda3 -> Anaconda Navigator)
    + In the upper tool bar, change “Applications on” from “base” or “anaconda” to “pg” (you will have to do this each time you open Anaconda!)
    + In the main window search for the icon “Spyder” and click on “Install” or, if it is already installed, on “Launch”.

6. Open Spyder
        + Open file refraPy.py (File -> open…)
        + In the Spyder tool bar click Run -> Configuration per file -> Execute in dedicated console
