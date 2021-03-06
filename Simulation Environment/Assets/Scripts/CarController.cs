using UnityEngine;
using UnityEngine.UI;
using System.Collections;


// This class is repsonsible for controlling inputs to the car.
[RequireComponent(typeof(Drivetrain))]
public class CarController : MonoBehaviour {

    // Add all wheels of the car here, so brake and steering forces can be applied to them.
    public Wheel[] wheels;

    // A transform object which marks the car's center of gravity.
    // Cars with a higher CoG tend to tilt more in corners.
    // The further the CoG is towards the rear of the car, the more the car tends to oversteer. 
    // If this is not set, the center of mass is calculated from the colliders.
    public Transform centerOfMass;

    // A factor applied to the car's inertia tensor. 
    // Unity calculates the inertia tensor based on the car's collider shape.
    // This factor lets you scale the tensor, in order to make the car more or less dynamic.
    // A higher inertia makes the car change direction slower, which can make it easier to respond to.
    public float inertiaFactor = 1.5f;

    // current input state
    [HideInInspector]
    public float brake;
    [HideInInspector]
    float throttle;
    float throttleInput;
    float clutch;
    [HideInInspector]
    public float steering;
    float lastShiftTime = -1;
    [HideInInspector]
    public float handbrake;

    // cached Drivetrain reference
    Drivetrain drivetrain;

    // How long the car takes to shift gears
    public float shiftSpeed = 0.8f;


    // These values determine how fast throttle value is changed when the accelerate keys are pressed or released.
    // Getting these right is important to make the car controllable, as keyboard input does not allow analogue input.
    // There are different values for when the wheels have full traction and when there are spinning, to implement 
    // traction control schemes.

    // How long it takes to fully engage the throttle
    public float throttleTime = 1.0f;
    // How long it takes to fully engage the throttle 
    // when the wheels are spinning (and traction control is disabled)
    public float throttleTimeTraction = 10.0f;
    // How long it takes to fully release the throttle
    public float throttleReleaseTime = 0.5f;
    // How long it takes to fully release the throttle 
    // when the wheels are spinning.
    public float throttleReleaseTimeTraction = 0.1f;

    // Turn traction control on or off
    public bool tractionControl = false;

    // Turn ABS control on or off
    public bool absControl = false;

    // These values determine how fast steering value is changed when the steering keys are pressed or released.
    // Getting these right is important to make the car controllable, as keyboard input does not allow analogue input.

    // How long it takes to fully turn the steering wheel from center to full lock
    public float steerTime = 1.2f;
    // This is added to steerTime per m/s of velocity, so steering is slower when the car is moving faster.
    public float veloSteerTime = 0.1f;

    // How long it takes to fully turn the steering wheel from full lock to center
    public float steerReleaseTime = 0.6f;
    // This is added to steerReleaseTime per m/s of velocity, so steering is slower when the car is moving faster.
    public float veloSteerReleaseTime = 0f;
    // When detecting a situation where the player tries to counter steer to correct an oversteer situation,
    // steering speed will be multiplied by the difference between optimal and current steering times this 
    // factor, to make the correction easier.
    public float steerCorrectionFactor = 10.0f;

    private bool gearShifted = false;
    private bool gearShiftedFlag = false;

    // Go right if true
    private bool goRight = false;
    // Go left if true
    private bool goLeft = false;
    // Accelerate if true
    private bool goStraight = false;
    // Brake if true
    private bool stop = false;

    private bool isManual = true;

    private bool is_done = false;

    private int stationary_updates = 0;

    public void set_done(bool done){
        this.is_done = done;
    }

    public bool get_done(){
        return this.is_done;
    }

    private void check_stationary(){
        if(System.Math.Round(GetComponent<Rigidbody>().velocity.magnitude, 2) < 0.1)
        {
            stationary_updates++;
            if(stationary_updates >= 100){
                stationary_updates = 0;
                resetPosition();
            }
        }
    }

    // Used by SoundController to get average slip velo of all wheels for skid sounds.
    public float slipVelo
    {
        get
        {
            float val = 0.0f;
            foreach (Wheel w in wheels)
                val += w.slipVelo / wheels.Length;
            return val;
        }

    }

    public void changeController()
    {
        isManual = !isManual;
        Debug.Log(isManual);
    }


    // Resets the car's position to a random checkpoint on the track.
    public void resetPosition()
    {
        GameObject checkpoint = Checkpoint.getRandomCheckpoint();
        print("Reset");
        GetComponent<Rigidbody>().velocity = Vector3.zero;
        GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        this.transform.position = checkpoint.transform.position;
        this.transform.rotation = checkpoint.transform.rotation;
        Checkpoint.setCurrentCheckPoint(checkpoint);
        is_done = true;
    }

    public void setAcceleration(string acceleration)
    {
        if (string.Compare(acceleration, "Acc.ACCELERATE") == 0)
        {
            goStraight = true;
            stop = false;
        }
        else if (string.Compare(acceleration, "Acc.BRAKE") == 0)
        {
            stop = true;
            goStraight = false;
        }
        else
        {
            goStraight = false;
            stop = false;
        }
    }

    public void setDirection(string direction)
    {
        if (string.Compare(direction, "Steer.TURN_LEFT") == 0)
        {
            goLeft = true;
            goRight = false;
        }
        else if (string.Compare(direction, "Steer.TURN_RIGHT") == 0)
        {
            goRight = true;
            goLeft = false;
        }
        else
        {
            goRight = false;
            goLeft = false;
        }
    }

    // Initialize
    void Start() {

        if (centerOfMass != null)
            GetComponent<Rigidbody>().centerOfMass = centerOfMass.localPosition;

        GetComponent<Rigidbody>().inertiaTensor *= inertiaFactor;
        drivetrain = GetComponent(typeof(Drivetrain)) as Drivetrain;
    }

    void Update() {
        check_stationary();

        Debug.Log(string.Format("goStraight: {0}, brake: {1}, left: {2}, right: {3}", goStraight, stop, goLeft, goRight));

        bool switchController = Input.GetKeyDown("o");


        if (switchController)
        {
            changeController();
        }

        if (Input.GetKeyDown("l")) 
        {
            resetPosition();
        }
        // Steering
        Vector3 carDir = transform.forward;
        float fVelo = GetComponent<Rigidbody>().velocity.magnitude;
        Vector3 veloDir = GetComponent<Rigidbody>().velocity * (1 / fVelo);
        float angle = -Mathf.Asin(Mathf.Clamp(Vector3.Cross(veloDir, carDir).y, -1, 1));
        float optimalSteering = angle / (wheels[0].maxSteeringAngle * Mathf.Deg2Rad);
        if (fVelo < 1)
            optimalSteering = 0;

        float steerInput = 0;

        // // Change this value to go left
        // if (Input.GetKey(KeyCode.LeftArrow))
        //     steerInput = -1;

        // // Change this value to go right
        // if (Input.GetKey(KeyCode.RightArrow))
        //     steerInput = 1;

        bool accelKey = false;
        bool brakeKey = false;

        if (isManual)
        {
            if (Input.GetKey(KeyCode.LeftArrow))
                steerInput = -1;
            if (Input.GetKey(KeyCode.RightArrow))
                steerInput = 1;
            
            accelKey = Input.GetKey(KeyCode.UpArrow);
            brakeKey = Input.GetKey(KeyCode.DownArrow);
        }
        else
        {
            if (goLeft)
                steerInput = -1;
            if (goRight)
                steerInput = 1;

            accelKey = goStraight;
            brakeKey = stop;
        }

        // Debug.Log(string.Format("goStraight: {0}, brake: {1}, direction: {2}", accelKey, brakeKey, steerInput));

        if (steerInput < steering) {
            float steerSpeed = (steering > 0) ? (1 / (steerReleaseTime + veloSteerReleaseTime * fVelo)) : (1 / (steerTime + veloSteerTime * fVelo));
            if (steering > optimalSteering)
                steerSpeed *= 1 + (steering - optimalSteering) * steerCorrectionFactor;
            steering -= steerSpeed * Time.deltaTime;
            if (steerInput > steering)
                steering = steerInput;
        } else if (steerInput > steering) {
            float steerSpeed = (steering < 0) ? (1 / (steerReleaseTime + veloSteerReleaseTime * fVelo)) : (1 / (steerTime + veloSteerTime * fVelo));
            if (steering < optimalSteering)
                steerSpeed *= 1 + (optimalSteering - steering) * steerCorrectionFactor;
            steering += steerSpeed * Time.deltaTime;
            if (steerInput < steering)
                steering = steerInput;
        }

        
        // // Change this value to see if the car must accelerate this frame or not.
        // bool accelKey = Input.GetKey(KeyCode.UpArrow);

        // // Change this value to see if the car must brake this frame or not.
        // bool brakeKey = Input.GetKey(KeyCode.DownArrow);

        // if (drivetrain.automatic && drivetrain.gear == 0) {
        //     accelKey = Input.GetKey(KeyCode.DownArrow);
        //     brakeKey = Input.GetKey(KeyCode.UpArrow);
        // }

        if (Input.GetKey(KeyCode.LeftShift)) {
            throttle += Time.deltaTime / throttleTime;
            throttleInput += Time.deltaTime / throttleTime;
        } else if (accelKey) {
            if (drivetrain.slipRatio < 0.10f)
                throttle += Time.deltaTime / throttleTime;
            else if (!tractionControl)
                throttle += Time.deltaTime / throttleTimeTraction;
            else
                throttle -= Time.deltaTime / throttleReleaseTime;

            if (throttleInput < 0)
                throttleInput = 0;
            throttleInput += Time.deltaTime / throttleTime;
        } else {
            if (drivetrain.slipRatio < 0.2f)
                throttle -= Time.deltaTime / throttleReleaseTime;
            else
                throttle -= Time.deltaTime / throttleReleaseTimeTraction;
        }

        throttle = Mathf.Clamp01(throttle);

        if (brakeKey) {
            if (drivetrain.slipRatio < 0.2f)
                brake += Time.deltaTime / throttleTime;
            else
                brake += Time.deltaTime / throttleTimeTraction;
            throttle = 0;
            throttleInput -= Time.deltaTime / throttleTime;
        } else {
            if (drivetrain.slipRatio < 0.2f)
                brake -= Time.deltaTime / throttleReleaseTime;
            else
                brake -= Time.deltaTime / throttleReleaseTimeTraction;
        }

        brake = Mathf.Clamp01(brake);
        throttleInput = Mathf.Clamp(throttleInput, -1, 1);

        // Handbrake
        handbrake = (Input.GetKey(KeyCode.Space) || Input.GetKey(KeyCode.JoystickButton2)) ? 1f : 0f;

        // Gear shifting
        float shiftThrottleFactor = Mathf.Clamp01((Time.time - lastShiftTime) / shiftSpeed);

        if (drivetrain.gear == 0 && Input.GetKey(KeyCode.UpArrow)) {
            throttle = 0.4f;// Anti reverse lock thingy??
        }
        
        if (drivetrain.gear == 0)
            drivetrain.throttle = Input.GetKey(KeyCode.UpArrow) ? throttle : 0f;
        else
            drivetrain.throttle = accelKey ? (tractionControl ? throttle : 1) * shiftThrottleFactor : 0f;
        // Debug.Log("Throttle " + drivetrain.throttle);
        drivetrain.throttleInput = throttleInput;

        if (Input.GetKeyDown(KeyCode.A)) {
            lastShiftTime = Time.time;
            drivetrain.ShiftUp();
        }

        if (Input.GetKeyDown(KeyCode.Z)) {
            lastShiftTime = Time.time;
            drivetrain.ShiftDown();
        }

        //play gear shift sound
        if (gearShifted && gearShiftedFlag && drivetrain.gear != 1) {
            GetComponent<SoundController>().playShiftUp();
            gearShifted = false;
            gearShiftedFlag = false;
        }


        // ABS Trigger (This prototype version is used to prevent wheel lock , currently expiremental)
        if (absControl)
            brake -= brake >= 0.1f ? 0.1f : 0f;

        // Apply inputs
        foreach (Wheel w in wheels) {
            w.brake = brakeKey ? brake :  0;
            w.handbrake = handbrake;
            w.steering = steering;
        }

        // Reset Car position and rotation in case it rolls over
        if (Input.GetKeyDown(KeyCode.R)) {
            transform.position = new Vector3(transform.position.x, transform.position.y + 2f, transform.position.z);
            transform.rotation = Quaternion.Euler(0, transform.localRotation.y, 0);
        }


        // Traction Control Toggle
        if (Input.GetKeyDown(KeyCode.T)) {

            if (tractionControl) {
                tractionControl = false;
            } else {
                tractionControl = true;
            }
        }

        // Anti-Brake Lock Toggle
        if (Input.GetKeyDown(KeyCode.B)) {
            if (absControl) {
                absControl = false;
            } else {
                absControl = true;
            }
        }
    }

    // Debug GUI. Disable when not needed.
    // void OnGUI() {
    //     GUI.Label(new Rect(0, 60, 100, 200), "km/h: " + GetComponent<Rigidbody>().velocity.magnitude * 3.6f);
    //     tractionControl = GUI.Toggle(new Rect(0, 80, 300, 20), tractionControl, "Traction Control (bypassed by shift key)");
    // }
}