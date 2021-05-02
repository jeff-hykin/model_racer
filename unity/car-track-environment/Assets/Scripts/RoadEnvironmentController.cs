using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class RoadEnvironmentController : MonoBehaviour
{
    private List<GameObject> obstacles = new List<GameObject>();
    private List<Vector3> originalObstaclePositions = new List<Vector3>();
    public float defaultBlockOffset = 5f;
    void Awake()
    {
        GetObstacleOriginalPosition();
    }
    private void GetObstacleOriginalPosition()
    {
        foreach (GameObject obstacle in GameObject.FindGameObjectsWithTag("obstaclex"))
        {
            obstacles.Add(obstacle);
            originalObstaclePositions.Add(obstacle.transform.position);
        }
        foreach (GameObject obstacle in GameObject.FindGameObjectsWithTag("obstacley"))  // TODO: Change naming to Z instead of Y
        {
            obstacles.Add(obstacle);
            originalObstaclePositions.Add(obstacle.transform.position);
        }
    }

    public void ResetArea()
    {
        MoveBlocks();
    }

    private void MoveBlocks()
    {
        var block_offset = Academy.Instance.EnvironmentParameters.GetWithDefault("block_offset", defaultBlockOffset);
        int i = 0;
        foreach (GameObject obstacle in GameObject.FindGameObjectsWithTag("obstaclex"))
        {
            var tmp_offset = block_offset;
            if (block_offset == 10.0f)
            {
                tmp_offset = Random.Range(0f, 12.0f);  // 12 to offset wall thickness
            }
            obstacle.transform.position = originalObstaclePositions[i] + new Vector3(tmp_offset, 0, 0);
            i++;
        }
        foreach (GameObject obstacle in GameObject.FindGameObjectsWithTag("obstacley"))  // TODO: Change naming to Z instead of Y
        {
            var tmp_offset = block_offset;
            if (block_offset == 10.0f)
            {
                tmp_offset = Random.Range(0f, 12.0f);
            }
            obstacle.transform.position = originalObstaclePositions[i] + new Vector3(0, 0, tmp_offset);
            i++;
        }
    }

    // Update is called once per frame
    // void Update()
    //{
    //   
    //}
}
