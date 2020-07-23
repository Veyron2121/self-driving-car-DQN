using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

[Serializable]
public class Information
{
    public double velocity;
    public double angle_from_road;
    public double distance_from_road;
    public string image_path;
    public bool is_done;

}

public class AgentInformation : MonoBehaviour
{

    public Text gui_info;


    private Information info;

    // private Checkpoint checkpoint_info;
    private GameObject player;


    // Start is called before the first frame update
    void Start()
    {
        info = new Information();
        info.velocity = 0;
        info.angle_from_road = 0;
        info.distance_from_road = 0;
        info.image_path = "";
        player = GameObject.FindGameObjectsWithTag("Player")[0];
    }

    void updateInfo()
    {
        info.velocity = System.Math.Round(player.GetComponent<Rigidbody>().velocity.magnitude, 2);

        Vector3 checkpoint_location = Checkpoint.getCurrentCheckpointLocation();
        Vector3 vector_to_checkpoint = checkpoint_location - player.transform.position;

        info.distance_from_road = System.Math.Round(vector_to_checkpoint.magnitude, 2);
        info.angle_from_road = System.Math.Round(Vector3.SignedAngle(vector_to_checkpoint, player.transform.forward, Vector3.up) * Mathf.Deg2Rad, 2);

        // info.image_path = Path.Combine(Application.dataPath, TakeScreenshot.takeScreenshot(400, 400));
        info.image_path = Application.dataPath + "/" + TakeScreenshot.takeScreenshot(32, 32);

        info.is_done = player.is_done;
    }

    void handleNextAction(string nextAction)
    {
        // Debug.Log(nextAction);
        string[] actions = nextAction.Split(',');
        player.GetComponent<CarController>().setAcceleration(actions[0]);
        player.GetComponent<CarController>().setDirection(actions[1]);
        if (info.is_done){
            player.is_done = false;
        }
    }


    // Update is called once per frame
    void Update()
    {
        updateInfo();
        // bool getAction = Input.GetKeyDown("o");
        bool getAction = true;
        if (getAction){
            string gameState = JsonUtility.ToJson(info);
            string nextAction = ActionRequester.getNextAction(gameState);
            handleNextAction(nextAction);
            getAction = false;
        }
        
        // Remove this line to disable GUI
        gui_info.text = "Speed: " + info.velocity + "\nDistance: " + info.distance_from_road + "\nAngle: " + info.angle_from_road;
        
    }
}
