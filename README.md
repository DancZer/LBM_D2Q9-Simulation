# LBM_D2Q9-Simulation
Hello everyone!

This was my BSc thesis I created in 2011. 
Since then new version of Qt and CUDA has been released. 
I adopted to the latest version of Cuda 11.6 and to Qt 5.15 my project.

No new features were introduced. Probably the architecture and the implementation should be improved, because the GPU is not utilized 100%.

Best regards,
DancZer



----------------------Setup-----------------------
Install CUDA SDK 11.6
Install Qt 5.15

In case you would like to use anothere CUDA version update the cuda location in .pro file.



----------------------Application-----------------------
You can edit and execute 2D simulation with the application. In the settings you can adjust the physical properties and the execution mode (CPU, GPU).
Back is the fluid area, colors could indicate different bounderies. White is the wall.


Things to do for simulation:
1. draw or open an existing mask (left click <selected color>, right click white)
2. add boundary conditions (in default case 2)
3. modify the boundary conditions (change pressure or x,y speeds) e.g: vx:10, for red and for green too.
4. press the play button

The Application first initialize the simulation, then switches to the first tab an display the execution progress.