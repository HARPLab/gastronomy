// ARDUINO MEGA PIN LAYOUT DEFINES
//#define 0
//#define 1
#define Delta1Motor1En 2 // PWM Pin
#define Delta1Motor2En 3 // PWM Pin
#define Delta1Motor3En 4 // PWM Pin
#define Delta2Motor1En 5 // PWM Pin
#define Delta2Motor2En 6 // PWM Pin
#define Delta2Motor3En 7 // PWM Pin
//#define 8 // PWM Pin
//#define 9 // PWM Pin
//#define 10 // PWM Pin
//#define 11 // PWM Pin
//#define 12 // PWM Pin
//#define 13 // PWM Pin
//#define 14
//#define 15
//#define 16
//#define 17
//#define 18 // Interrupt Pin
//#define 19 // Interrupt Pin
//#define 20 // Interrupt Pin, SDA
//#define 21 // Interrupt Pin, SCL
#define Delta1Motor1In1 22
#define Delta1Motor1In2 23
#define Delta1Motor2In1 24
#define Delta1Motor2In2 25
#define Delta1Motor3In1 26
#define Delta1Motor3In2 27
#define Delta2Motor1In1 28
#define Delta2Motor1In2 29
#define Delta2Motor2In1 30
#define Delta2Motor2In2 31
#define Delta2Motor3In1 32
#define Delta2Motor3In2 33
//#define 34 
//#define 35
//#define 36
//#define 37
//#define 38
//#define 39
//#define 40
//#define 41
//#define 42
//#define 43
//#define 44
//#define 45
//#define 46
//#define 47
//#define 48
//#define 49
//#define 50
//#define 51
//#define 52
//#define 53

// ROS INCLUDES
#include <ros.h> // ROS Serial
#include <ros/time.h> // ROS time for publishing messages
#include <tf/transform_broadcaster.h> // Transform Broadcaster for TF
#include <std_msgs/Empty.h> // Empty msg for Homing everything
#include <std_msgs/Bool.h> // Bool msg for Emergency stop
#include <delta_msgs/Float32Array.h> // Float32Array msg for Trajectory
#include <sensor_msgs/JointState.h> // JointState msg for publishing the robot's current joint values

// ROS Globals
ros::NodeHandle nh;

// // TF Broadcaster Globals
// geometry_msgs::TransformStamped t;
// char base_link[] = "/base_link";

// // TF Broadcaster
// tf::TransformBroadcaster broadcaster;

sensor_msgs::JointState joint_state_msg;
ros::Publisher joint_state_pub("/delta_array/joint_state", &joint_state_msg);
char *joint_names[] = {"delta1_motor1", "delta1_motor2", 
                       "delta1_motor3", "delta2_motor1", 
                       "delta2_motor2", "delta2_motor3"};
float joint_positions[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
float joint_velocities[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

int delta_en[6] = {Delta1Motor1En, Delta1Motor2En, 
                   Delta1Motor3En, Delta2Motor1En, 
                   Delta2Motor2En, Delta2Motor3En};

int delta_in1[6] = {Delta1Motor1In1, Delta1Motor2In1, 
                    Delta1Motor3In1, Delta2Motor1In1, 
                    Delta2Motor2In1, Delta2Motor3In1};

int delta_in2[6] = {Delta1Motor1In2, Delta1Motor2In2, 
                    Delta1Motor3In2, Delta2Motor1In2, 
                    Delta2Motor2In2, Delta2Motor3In2};

// STOP COMMAND
// Stop Command Globals
bool stop_flag = false;

// Stop Command Callback
// Immediately halts everything by setting all the motor control pins to LOW and sets the stop_flag to true.
// Remember to publish a false msg to restart the robot.
void stopCallback(const std_msgs::Bool& stop_msg){

  // If the stop_msg contains true, stop all of the motors
  if(stop_msg.data)
  {
    // set stop_flag to true
    stop_flag = true;

    // Turn off motors
    for(int i = 0; i < 6; i++)
    {
      analogWrite(delta_en[i], 0);
      digitalWrite(delta_in1[i], LOW);
      digitalWrite(delta_in2[i], LOW);
    }
  }
  // Otherwise if the stop_msg contains false, set the stop_flag back to false
  // and enable the stepper motors.
  else
  {
    stop_flag = false;
  }
}

// Stop Command Subscriber
ros::Subscriber<std_msgs::Bool> stop_sub("/delta_array/stop", &stopCallback);

// Desired Delta Positions and Velocities (hard cap of 8 trajectory points)
float desired_joint_positions[8][6];
float desired_joint_velocities[8][6];
float durations[8];
float last_time = 0.0;
int num_trajectory_points = 0;
int current_trajectory_point = 0;
bool position_trajectory = false;
bool velocity_trajectory = false;

// Done Moving Deltas Publisher
std_msgs::Bool done_moving_deltas_msg;
ros::Publisher done_moving_deltas_pub("/delta_array/done_moving_deltas", &done_moving_deltas_msg);

// Delta Position Trajectory Callback
// Stores the Delta Position Trajectory to desired_joint_positions
void deltaPositionTrajectoryCallback(const delta_msgs::Float32Array& joint_trajectory_msg)
{
  num_trajectory_points = joint_trajectory_msg.data_length / 6;

  for(int i = 0; i < num_trajectory_points; i++)
  {
    for(int j = 0; j < 6; j++)
    {
      desired_joint_positions[i][j] = joint_trajectory_msg.data[6*i+j];
    }
  }

  position_trajectory = true;
  velocity_trajectory = false;
  current_trajectory_point = 0;
}

// Delta Position Trajectory Subscriber
ros::Subscriber<delta_msgs::Float32Array> delta_position_trajectory_sub("/delta_array/joint_position_trajectory", &deltaPositionTrajectoryCallback);

// Delta Velocity Trajectory Callback
// Stores the Delta Velocity Trajectory to desired_joint_velocities
void deltaVelocityTrajectoryCallback(const delta_msgs::Float32Array& joint_trajectory_msg)
{
  num_trajectory_points = joint_trajectory_msg.data_length / 7;

  for(int i = 0; i < num_trajectory_points; i++)
  {
    for(int j = 0; j < 6; j++)
    {
      desired_joint_velocities[i][j] = joint_trajectory_msg.data[7*i+j];
    }
    durations[i] = joint_trajectory_msg.data[7*i+6];
  }

  position_trajectory = false;
  velocity_trajectory = true;
  current_trajectory_point = 0;
  last_time = 0.0;
}

// Delta Velocity Trajectory Subscriber
ros::Subscriber<delta_msgs::Float32Array> delta_velocity_trajectory_sub("/delta_array/joint_velocity_trajectory", &deltaVelocityTrajectoryCallback);

float max_motor_speed[6] = {0.005, 0.005, 0.005, 0.005, 0.005, 0.005};

bool reset = false;

// RESET COMMAND
// Reset Command Callback
// Resets the robot's desired positions to the default positions.
void resetCallback(const std_msgs::Empty& reset_msg){

  num_trajectory_points = 1;
  durations[0] = 11.0;
  for(int i = 0; i < 6; i++)
  {
    desired_joint_velocities[0][i] = -max_motor_speed[i]/2;
  }

  position_trajectory = false;
  velocity_trajectory = true;
  current_trajectory_point = 0;
  last_time = 0.0;
  reset = true;
}

// Reset Command Subscriber
ros::Subscriber<std_msgs::Empty> reset_sub("/delta_array/reset", &resetCallback);

unsigned long current_arduino_time;
unsigned long last_arduino_time;
float time_elapsed;

// SETUP CODE
void setup()
{ 
  // set all the base dc motor control pins to outputs
  for(int i = 0; i < 6; i++)
  {
    pinMode(delta_en[i], OUTPUT);
    pinMode(delta_in1[i], OUTPUT);
    pinMode(delta_in2[i], OUTPUT);
  }

  // disable dc motors by setting their enable lines to low
  for(int i = 0; i < 6; i++)
  {
    analogWrite(delta_en[i], 0);
    digitalWrite(delta_in1[i], LOW);
    digitalWrite(delta_in2[i], LOW);
  }

  // ROS Serial Initialization Code
  // Initialize Node
  nh.initNode();

  //nh.getHardware()->setBaud(86400);

  // // Initialize tf broadcaster
  // broadcaster.init(nh);
  
  // Advertise topics
  nh.advertise(joint_state_pub);
  nh.advertise(done_moving_deltas_pub);

  // Subscribe to topics
  nh.subscribe(stop_sub);
  nh.subscribe(reset_sub);
  nh.subscribe(delta_position_trajectory_sub);
  nh.subscribe(delta_velocity_trajectory_sub);
  
  // Joint State Msg Setup Code
  joint_state_msg.name_length = 6;
  joint_state_msg.velocity_length = 6;
  joint_state_msg.position_length = 6;
  joint_state_msg.effort_length = 0;
  joint_state_msg.name = joint_names;
}

void checkIfDoneMovingDeltas()
{
  if(current_trajectory_point > 0 && current_trajectory_point == num_trajectory_points)
  {
    position_trajectory = false;
    velocity_trajectory = false;
    current_trajectory_point = 0;
    num_trajectory_points = 0;
    last_time = 0.0;
    done_moving_deltas_msg.data = true;
    done_moving_deltas_pub.publish(&done_moving_deltas_msg);
    done_moving_deltas_pub.publish(&done_moving_deltas_msg);
    if(reset)
    {
      reset = false;
      for(int i = 0; i < 6; i++)
      {
        joint_positions[i] = 0.0;
        joint_velocities[i] = 0.0;
      }
    }
  }
}

// LOOP CODE
void loop()
{
  // If the robot is not currently in the stop mode
  if(!stop_flag)
  {
      
    if(position_trajectory)
    {
      moveDeltaPosition();
    }
    else if(velocity_trajectory)
    {
      moveDeltaVelocity();
    }
    checkIfDoneMovingDeltas();
  }
  
  publishJointStates();
  // updateTransform();
  // publishTransform();

  nh.spinOnce();
  //delay(5);
  current_arduino_time = millis();
  time_elapsed = float(current_arduino_time - last_arduino_time) / 1000.0;
  last_time += time_elapsed;
  for(int i = 0; i < 6; i++)
  {
    joint_positions[i] += joint_velocities[i] * time_elapsed;
  }
  last_arduino_time = current_arduino_time;
}

float position_threshold = 0.001;
float joint_errors[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

void moveDeltaPosition()
{
  bool reached_point = true;
  for(int i = 0; i < 6; i++)
  {
    joint_errors[i] = joint_positions[i] - desired_joint_positions[current_trajectory_point][i];
    if(fabs(joint_errors[i]) > position_threshold)
    {
      reached_point = false;
    }
  }
  if(reached_point)
  {
    current_trajectory_point += 1;
  }
  if(current_trajectory_point < num_trajectory_points)
  {
    for(int i = 0; i < 6; i++)
    {
      if(joint_errors[i] > position_threshold)
      {
        //analogWrite(delta_en[i], 255);
        digitalWrite(delta_en[i], HIGH);
        digitalWrite(delta_in1[i], LOW);
        digitalWrite(delta_in2[i], HIGH);
        joint_velocities[i] = -max_motor_speed[i];
      }
      else if(joint_errors[i] < position_threshold)
      {
        //analogWrite(delta_en[i], 255);
        digitalWrite(delta_en[i], HIGH);
        digitalWrite(delta_in1[i], HIGH);
        digitalWrite(delta_in2[i], LOW);
        joint_velocities[i] = max_motor_speed[i];
      }
      else
      {
        //analogWrite(delta_en[i], 0);
        digitalWrite(delta_en[i], LOW);
        digitalWrite(delta_in1[i], LOW);
        digitalWrite(delta_in2[i], LOW);
        joint_velocities[i] = 0.0;
      }
    }
  }
  else
  {
    for(int i = 0; i < 6; i++)
    {
      //analogWrite(delta_en[i], 0);
      digitalWrite(delta_en[i], LOW);
      digitalWrite(delta_in1[i], LOW);
      digitalWrite(delta_in2[i], LOW);
      joint_velocities[i] = 0.0;
    }
  }
}

void moveDeltaVelocity()
{
  if(last_time >= durations[current_trajectory_point])
  {
    current_trajectory_point += 1;
  } 
  if(current_trajectory_point < num_trajectory_points)
  {
    for(int i = 0; i < 6; i++)
    {
      joint_velocities[i] = desired_joint_velocities[current_trajectory_point][i];
      if(joint_velocities[i] < 0.0)
      {
        //int motor_speed = (int)(max((-joint_velocities[i]/ max_motor_speed[i]), 1.0) * 255.0);
        //analogWrite(delta_en[i], motor_speed);
        digitalWrite(delta_en[i], HIGH);
        digitalWrite(delta_in1[i], LOW);
        digitalWrite(delta_in2[i], HIGH);
      }
      else if(joint_velocities[i] > 0.0)
      {
        //int motor_speed = (int)(max((joint_velocities[i]/ max_motor_speed[i]), 1.0) * 255.0);
        //analogWrite(delta_en[i], motor_speed);
        digitalWrite(delta_en[i], HIGH);
        digitalWrite(delta_in1[i], HIGH);
        digitalWrite(delta_in2[i], LOW);
      }
      else
      {
        //analogWrite(delta_en[i], 0);
        digitalWrite(delta_en[i], LOW);
        digitalWrite(delta_in1[i], LOW);
        digitalWrite(delta_in2[i], LOW);
      }
    }
  }
  else
  {
    for(int i = 0; i < 6; i++)
    {
      //analogWrite(delta_en[i], 0);
      digitalWrite(delta_en[i], LOW);
      digitalWrite(delta_in1[i], LOW);
      digitalWrite(delta_in2[i], LOW);
      joint_velocities[i] = 0.0;
      if(joint_positions[i] < 0.0)
      {
        joint_positions[i] = 0.0;
      }
    }
  }
}

void publishJointStates()
{
  joint_state_msg.header.stamp = nh.now();

  joint_state_msg.position = joint_positions;
  joint_state_msg.velocity = joint_velocities;
  
  joint_state_pub.publish(&joint_state_msg);
}

// void updateTransform()
// {

//   current_x_position = (front_motor_encoder_count / (float)encoder_counts_per_revolution) * distance_traveled_per_wheel_revolution;
//   current_y_position = (left_motor_encoder_count / (float)encoder_counts_per_revolution) * distance_traveled_per_wheel_revolution;
//   previous_x_positions[x_save_index] = current_x_position;
//   x_save_index = (x_save_index + 1) % 10;
//   previous_y_positions[y_save_index] = current_y_position;
//   y_save_index = (y_save_index + 1) % 10;
  
//   t.header.frame_id = odom;
//   t.child_frame_id = base_link;
//   t.transform.translation.x = current_x_position; 
//   t.transform.translation.y = current_y_position;
//   t.transform.translation.z = 1.0;  
//   t.transform.rotation.x = (back_motor_encoder_count / (float)encoder_counts_per_revolution) * distance_traveled_per_wheel_revolution;
//   t.transform.rotation.y = (right_motor_encoder_count / (float)encoder_counts_per_revolution) * distance_traveled_per_wheel_revolution; 
//   t.transform.rotation.z = 0.0; 
//   t.transform.rotation.w = 1.0;  
//   t.header.stamp = nh.now();
// }

// void publishTransform()
// {
//   broadcaster.sendTransform(t);
// }
