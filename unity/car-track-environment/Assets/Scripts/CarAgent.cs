﻿using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class CarAgent : Agent
{
    public float speed = 10f;
    public float torque = 10f;
    public int score = 0;
    public bool resetOnCollision = true;
    public GameObject environmentObject;
    private Transform _track;
    private int actionCount = 0;
    private int maxActionCountPerEpisode = 10000;
    public Vector3 lastPos;
    public override void Initialize()
    {
        GetTrackIncrement();
    }

    private void MoveCar(float horizontal, float vertical, float dt)
    {
        float distance = speed * vertical;
        transform.Translate(distance * dt * Vector3.forward);

        float rotation = horizontal * torque * 90f;
        transform.Rotate(0f, rotation * dt, 0f);
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        float horizontal = vectorAction[0];
        float vertical = vectorAction[1];

        
        MoveCar(horizontal, vertical, Time.fixedDeltaTime);

        int reward = GetTrackIncrement();

        var movevec = transform.position - lastPos;
        float angle = Vector3.Angle(movevec, _track.forward);
        //float bonus = (1f - angle / 90f) * Mathf.clamp01(vertical) * time.fixeddeltatime;
        float bonus = 0;
        if (vertical < 0)
        {
            bonus =  -1* ((1f - angle / 120f) * Mathf.Clamp01(vertical)) * Time.fixedDeltaTime;  // slightly discorage moving backwards
        }
        else
        {
            bonus = ((1f - angle / 120f) * Mathf.Clamp01(vertical)) * Time.fixedDeltaTime;
        }
        //float bonus = vertical * Time.fixedDeltaTime;
        //if (vertical < 0)
        //{
        //    bonus = -1 * Time.fixedDeltaTime;
        //}
        AddReward(bonus);

        score += reward;
        actionCount++;
        if (actionCount >= maxActionCountPerEpisode)
        {
            actionCount = 0;
            EndEpisode();
        }
        lastPos = transform.position;
    }

    public override void Heuristic(float[] action)
    {
        action[0] = Input.GetAxis("Horizontal");
        action[1] = Input.GetAxis("Vertical");
    }

    public override void CollectObservations(VectorSensor vectorSensor)
    {
        float angle = Vector3.SignedAngle(_track.forward, transform.forward, Vector3.up);

    //    vectorSensor.AddObservation(angle / 180f);
    //    vectorSensor.AddObservation(ObserveRay(1.5f, .5f, 25f));
    //    vectorSensor.AddObservation(ObserveRay(1.5f, .375f, 20f));
    //    vectorSensor.AddObservation(ObserveRay(1.5f, .25f, 15f));
    //    vectorSensor.AddObservation(ObserveRay(1.5f, .125f, 10f));
    //    vectorSensor.AddObservation(ObserveRay(1.5f, 0f, 0f));
    //    vectorSensor.AddObservation(ObserveRay(1.5f, -.125f, -10f));
    //    vectorSensor.AddObservation(ObserveRay(1.5f, -.25f, -15f));
    //    vectorSensor.AddObservation(ObserveRay(1.5f, -.375f, -20f));
    //    vectorSensor.AddObservation(ObserveRay(1.5f, -.5f, -25f));
    //    vectorSensor.AddObservation(ObserveRay(-1.5f, 0, 180f));
    }

    private float ObserveRay(float z, float x, float angle)
    {
        var tf = transform;
    
        // Get the start position of the ray
        var raySource = tf.position + Vector3.up / 2f; 
        const float RAY_DIST = 5f;
        var position = raySource + tf.forward * z + tf.right * x;

        // Get the angle of the ray
        var eulerAngle = Quaternion.Euler(0, angle, 0f);
        var dir = eulerAngle * tf.forward;
    
        // See if there is a hit in the given direction
        Physics.Raycast(position, dir, out var hit, RAY_DIST);
        return hit.distance >= 0 ? hit.distance / RAY_DIST : -1f;
    }


    private int GetTrackIncrement()
    {
        int reward = 0;
        var carCenter = transform.position + Vector3.up;

        // Find what tile I'm on
        if (Physics.Raycast(carCenter, Vector3.down, out var hit, 2f))
        {
            var newHit = hit.transform;
            // Check if the tile has changed
            if (_track != null && newHit != _track)
            {
                float angle = Vector3.Angle(_track.forward, newHit.position - _track.position);
                reward = (angle < 90f) ? 1 : -1;
            }

            _track = newHit;
        }

        return reward;
    }

    public override void OnEpisodeBegin()
    {
        if (resetOnCollision)
        {
            transform.localPosition = Vector3.zero;
            transform.localRotation = Quaternion.identity;
        }
        environmentObject.GetComponent<RoadEnvironmentController>().ResetArea();
        lastPos = transform.position;
    }

    private void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("wall")|| other.gameObject.CompareTag("obstaclex")|| other.gameObject.CompareTag("obstacley") || other.gameObject.CompareTag("obstacle"))
        {
            SetReward(-1f);
            Debug.Log("Hit"+other.gameObject.name);
            EndEpisode();
        }
    }
}