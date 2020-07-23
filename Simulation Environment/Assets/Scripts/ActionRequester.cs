using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;


public class ActionRequester
{
    public static string getNextAction(string gameState)
    {
        ForceDotNet.Force();
        string nextAction = "";
        using(RequestSocket client = new RequestSocket())
        {
            client.Connect("tcp://localhost:5555");

            client.SendFrame(gameState);     
            nextAction = client.ReceiveFrameString();
            
            // Debug.Log("Received: " + nextAction);
        }

        NetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yet
        return nextAction;
    }
}
