# quadruped-rl

Implementation of a data-based control architecture on the Solo-12 quadruped, using neural networks trained with reinforcement learning.

# Training

See [https://github.com/Gepetto/soloRL] for more details about the training environment.

# Dependencies
* PLUS GAB: conda create --name solo12 python=3.8
* PLUS GAB: conda install conda-forge::matplotlib
* PLUS GAB:  conda install pinocchio -c conda-forge 
* PLUS GAB:  pip3 install numpy==1.23

* Install a Python 3 version (for instance Python 3.8)

* Install PyBullet (simulation environment): `pip3 install --user pybullet`

* Install Inputs (to control the robot with a gamepad): `pip3 install --user inputs`

* Clone `git clone --recursive https://gitlab.laas.fr/paleziart/quadruped-rl.git`

* Compile the C++ part that runs the neural network:
    * `cd cpuMLP`
    * `mkdir build`
    * `cd build`
    * `cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DPYTHON_EXECUTABLE=$(which python3) -DPYTHON_STANDARD_LAYOUT=ON`
    * `make`

* Run the controller in simulation: `python3 main_solo12_RL.py` 

* You can make the simulation longer by tweaking the `max_steps` of `RLParams` in Params.py. The control loop runs at 1 kHz so the number is in milliseconds.

* You can disable the plots at the end of the simulation by setting to False the variable `PLOTTING` of `RLParams` in Params.py.

# Using a joystick

* You can control the robot with your gamepad turning `USE_PREDEFINED` to False and `USE_JOYSTICK` to True.

* You will likely have to tweak the following formulas in Joystick.py since the values returned for each axes depends on the model of your gamepad.
```
self.vX = (self.gp.leftJoystickX.value / 0.00390625 - 1 ) * self.VxScale
self.vY = (self.gp.leftJoystickY.value / 0.00390625 - 1 ) * self.VyScale
self.vYaw = (self.gp.rightJoystickX.value / 0.00390625 - 1 ) * self.vYawScale
```

# Deployment on a real Solo-12

* TODO

