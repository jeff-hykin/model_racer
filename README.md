### Setup

1. Download or clone this repository (Note: it is currently quite large)<br>
    `git clone --branch unity https://github.com/jeff-hykin/model_racer.git`
    
2. Download Unity Hub (from an App Store, Package manger, or Unity's website)
    - agree to the license thing (green arrow)
    - press back (yellow arrow)
    <img src="/documentation/images/activate.png" alt="where-to-click">
    
    - go to projects (should be empty)
    <img src="/documentation/images/unity_hub.png" alt="where-to-click">

    - click the "Add" then open the `OpenMeWithUnity` folder from this repository (e.g. `model_racer/OpenMeWithUnity`)
    - Then click the name of the project (OpenMeWithUnity), and there should be a black pop-up (at the bottom) like this
    <img src="/documentation/images/install_prompt.png" alt="where-to-click">
    
    - Click install, then try opening the project again
3. Opening up the scene
    - You probably won't see the blue guy yet, but the menus should look something like
    <img src="/documentation/images/basic_run.png" alt="where-to-click">

    - To see the blue guy
      - (red arrow) open the "Assets/ML-Agents/Examples/3DBall/Scenes" folder 
      - (blue arrow) Then open the "3DBall" Scene 
      - (don't do the green arrow yet)
4. With your favorite text editor open the `OpenMeInTextEditor/main.py` file (this should look familiar as a stanard python RL setup)
5. Follow the setup instructions in `./documentation/SETUP.md`
6. Inside the shell (from the setup.md) run `run main`
7. Now click the green arrow inside Unity
<img src="/documentation/images/basic_run.png" alt="where-to-click">

8. (you should see output inside the shell now!)
