using UnityEngine;
using System.Collections;

 
public class TakeScreenshot : MonoBehaviour {
    public int resWidth = 2550; 
    public int resHeight = 3300;
    private bool takeSC = false;

    public static string ScreenShotName(int width, int height) 
    {
        return string.Format("Screenshots/screen_{0}.png", 
                             System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
    }

    public static void takeScreenshot(int i) 
    {
       // string name = ScreenShotName(width, height);
       string name = string.Format("Assets/Screenshots/PerspectiveSegment_{0}.png", i);
       ScreenCapture.CaptureScreenshot(name);
      //  print(name);
    }

    void LateUpdate() 
    {
       takeSC |= Input.GetKeyDown("p");
       if (takeSC)
       {
           ScreenCapture.CaptureScreenshot(ScreenShotName(resWidth, resHeight));
           takeSC = false;
       }
    }
         
 }