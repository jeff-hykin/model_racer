### Setup

1. Setup Unity `documentation/unity_setup.md`
2. Setup the python code `documentation/basic_python_setup.md`
3. With your favorite text editor open the `source/demo.py` file (this should look familiar as a stanard python RL setup)
4. Run `python ./source/demo.py`<br>you should see <br>`[INFO] Listening on port 5004. Start training by pressing the Play button in the Unity Editor.`
5. Now, go back to the Unity App and click the start button
<img src="/documentation/images/car_track_scene_ready_to_start.png" alt="description">

6. (you should see output inside the terminal now!)
7. If you want to modify the camera resolution, follow click the carcam (lime green), then go to the inspector, and scroll down until you see the camera settings.
<img src="/documentation/images/camera_sensor.png" alt="description">

8. If you want to run the JIRL code instead of the demo code<br> `cd ./source/jirl`<br> `python train.py`<br> (and dont forget to click start in Unity)

### Unity Curriculun Learning
Check `master-wip` branch for unity driving simulation and curriculum learning models and statistics.<p>
To see the trained models, refer to `model_racer/unity/car-track-environment/results`. `test0502_3DRaySimple` is trained without curriculum learning, and `test0502_3DRayCurriculum` is trained with curriculum learning. <p>
To check the training environment, go to `model_racer/unity/car-track-environment/Assets/Scenes/`. Use `CarTrainSimpleRay_Cylinder.unity` for car agent with ray sensor. Car with camera sensor is also available under prefab.<p>
Training configurations are located under `model_racer/unity/car-track-environment/TrainerConfigs/`<p>
