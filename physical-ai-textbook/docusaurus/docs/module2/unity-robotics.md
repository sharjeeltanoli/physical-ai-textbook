---
sidebar_position: 11
title: Unity for Robotics
description: Leveraging Unity game engine for photorealistic robot simulation
---

# Unity for Robotics

## Introduction

Unity is a powerful game engine that can be adapted for robotics simulation, offering photorealistic graphics, efficient rendering, and a vast asset ecosystem. This chapter covers integrating Unity with ROS 2 for robotics applications.

## Why Unity for Robotics?

### Advantages

**Visual Quality:**
- Photorealistic rendering
- Advanced lighting and shadows
- Post-processing effects
- Asset store with 3D models

**Performance:**
- Optimized game engine
- Cross-platform (Windows, Linux, macOS)
- Mobile and VR/AR support

**Development:**
- User-friendly editor
- Visual scripting
- Large community

### Limitations

- Not designed specifically for robotics
- Physics less accurate than dedicated simulators
- ROS integration requires setup

## Unity Robotics Hub

### Architecture

```
ROS 2 System                    Unity Simulation
┌─────────────┐                 ┌──────────────┐
│ ROS 2 Nodes │ ←──────────────→│ Unity Scene  │
│             │   TCP/UDP        │              │
│ Publishers  │   Connections    │ Robot Model  │
│ Subscribers │                  │ Environment  │
└─────────────┘                  └──────────────┘
```

### Components

1. **ROS-TCP-Connector** (Unity package)
2. **ROS-TCP-Endpoint** (ROS 2 package)
3. **URDF Importer** (Unity package)

## Installation

### Prerequisites

```bash
# Unity Hub (download from unity.com)
# Unity Editor 2021.3 LTS or later
# ROS 2 Humble
```

### Install ROS-TCP-Endpoint

```bash
cd ~/ros2_ws/src
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### Install Unity Packages

1. Open Unity Hub → Create New Project (3D)
2. Window → Package Manager
3. Add package from git URL:
   - `https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector`
   - `https://github.com/Unity-Technologies/URDF-Importer.git?path=/com.unity.robotics.urdf-importer`

## Importing Robot Models

### URDF to Unity

```csharp
// In Unity Editor:
// Assets → Import Robot from URDF
// Select your robot.urdf file
// Configure import settings:
// - Axis: Y-up
// - Mesh Decomposer: VHACD
// - Import
```

### Manual Configuration

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class RobotController : MonoBehaviour
{
    void Start()
    {
        // Connect to ROS
        ROSConnection.GetOrCreateInstance().Connect();
    }
}
```

## ROS 2 Integration

### Publishing from Unity

```csharp
using RosMessageTypes.Geometry;
using Unity.Robotics.ROSTCPConnector;

public class OdometryPublisher : MonoBehaviour
{
    ROSConnection ros;
    public string topicName = "odom";
    public float publishRate = 10f;

    private float timeElapsed;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistMsg>(topicName);
    }

    void Update()
    {
        timeElapsed += Time.deltaTime;

        if (timeElapsed > 1f / publishRate)
        {
            // Create message
            TwistMsg odomMsg = new TwistMsg
            {
                linear = new Vector3Msg
                {
                    x = transform.position.x,
                    y = transform.position.y,
                    z = transform.position.z
                },
                angular = new Vector3Msg
                {
                    x = 0, y = 0, z = 0
                }
            };

            // Publish
            ros.Publish(topicName, odomMsg);
            timeElapsed = 0;
        }
    }
}
```

### Subscribing in Unity

```csharp
using RosMessageTypes.Geometry;
using Unity.Robotics.ROSTCPConnector;

public class VelocitySubscriber : MonoBehaviour
{
    public string topicName = "cmd_vel";
    public float speed = 1.0f;

    void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<TwistMsg>(
            topicName,
            ReceiveVelocity
        );
    }

    void ReceiveVelocity(TwistMsg velocityMsg)
    {
        // Apply velocity to robot
        float linearX = (float)velocityMsg.linear.x * speed;
        float angularZ = (float)velocityMsg.angular.z;

        // Update robot position/rotation
        transform.Translate(Vector3.forward * linearX * Time.deltaTime);
        transform.Rotate(Vector3.up, angularZ * Time.deltaTime * Mathf.Rad2Deg);
    }
}
```

## Camera and Sensor Setup

### RGB Camera

```csharp
using UnityEngine;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector;

public class CameraPublisher : MonoBehaviour
{
    public Camera sensorCamera;
    public string topicName = "camera/image_raw";
    public int resolutionWidth = 640;
    public int resolutionHeight = 480;
    public float publishRate = 30f;

    private ROSConnection ros;
    private RenderTexture renderTexture;
    private Texture2D texture2D;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(topicName);

        renderTexture = new RenderTexture(
            resolutionWidth, resolutionHeight, 24
        );
        sensorCamera.targetTexture = renderTexture;

        texture2D = new Texture2D(
            resolutionWidth, resolutionHeight, TextureFormat.RGB24, false
        );

        InvokeRepeating("PublishImage", 0f, 1f / publishRate);
    }

    void PublishImage()
    {
        sensorCamera.Render();

        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(
            new Rect(0, 0, resolutionWidth, resolutionHeight), 0, 0
        );
        texture2D.Apply();

        ImageMsg imageMsg = new ImageMsg
        {
            header = new RosMessageTypes.Std.HeaderMsg(),
            height = (uint)resolutionHeight,
            width = (uint)resolutionWidth,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(resolutionWidth * 3),
            data = texture2D.GetRawTextureData()
        };

        ros.Publish(topicName, imageMsg);
    }
}
```

### Depth Camera

```csharp
public class DepthCameraPublisher : MonoBehaviour
{
    public Camera depthCamera;
    public Shader depthShader;

    void Start()
    {
        depthCamera.depthTextureMode = DepthTextureMode.Depth;
        depthCamera.SetReplacementShader(depthShader, "");
    }

    // Publish depth similar to RGB camera
}
```

## Building Environments

### Scene Setup

```csharp
// Create ground plane
GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
ground.transform.localScale = new Vector3(10, 1, 10);

// Add physics material
PhysicMaterial groundMaterial = new PhysicMaterial();
groundMaterial.staticFriction = 0.8f;
groundMaterial.dynamicFriction = 0.6f;
ground.GetComponent<Collider>().material = groundMaterial;

// Add obstacles
GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
obstacle.transform.position = new Vector3(3, 0.5f, 3);
obstacle.AddComponent<Rigidbody>().isKinematic = true;
```

### Lighting

```csharp
// Directional light (sun)
GameObject sunLight = new GameObject("Sun");
Light light = sunLight.AddComponent<Light>();
light.type = LightType.Directional;
light.intensity = 1.0f;
light.shadows = LightShadows.Soft;
sunLight.transform.rotation = Quaternion.Euler(50, -30, 0);
```

## Synthetic Data Generation

### Domain Randomization

```csharp
public class DomainRandomizer : MonoBehaviour
{
    public Material[] materials;
    public Light[] lights;

    void RandomizeScene()
    {
        // Randomize object textures
        foreach (GameObject obj in FindObjectsOfType<GameObject>())
        {
            Renderer renderer = obj.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material = materials[Random.Range(0, materials.Length)];
            }
        }

        // Randomize lighting
        foreach (Light light in lights)
        {
            light.intensity = Random.Range(0.5f, 2.0f);
            light.color = new Color(
                Random.Range(0.8f, 1.0f),
                Random.Range(0.8f, 1.0f),
                Random.Range(0.8f, 1.0f)
            );
        }
    }
}
```

## Launch Configuration

### Start ROS-TCP-Endpoint

```bash
# In ROS 2 workspace
ros2 run ros_tcp_endpoint default_server_endpoint \
  --ros-args -p ROS_IP:=127.0.0.1
```

### Unity Settings

```csharp
// In Unity: Robotics → ROS Settings
// ROS IP Address: 127.0.0.1
// ROS Port: 10000
// Protocol: ROS 2
```

## Best Practices

### Performance

- **Occlusion Culling**: Enable for large scenes
- **LOD (Level of Detail)**: Use for distant objects
- **Batching**: Combine static meshes

### Accuracy

- **Fixed Timestep**: Set consistent physics updates
  ```csharp
  Time.fixedDeltaTime = 0.02f; // 50Hz
  ```
- **Collision Detection**: Use Continuous for fast objects

### Debugging

```csharp
void OnDrawGizmos()
{
    // Visualize sensor ranges
    Gizmos.color = Color.red;
    Gizmos.DrawWireSphere(transform.position, sensorRange);
}
```

## Key Takeaways

- Unity provides photorealistic visualization for robotics
- ROS-TCP-Connector enables ROS 2 integration
- URDF importer simplifies robot model import
- Cameras and sensors publishable to ROS 2 topics
- Domain randomization enables robust vision models
- Performance optimization critical for real-time simulation

---

**Previous:** [Gazebo Simulator Fundamentals](./gazebo-fundamentals)
**Next:** [Simulation Best Practices](./simulation-best-practices)
