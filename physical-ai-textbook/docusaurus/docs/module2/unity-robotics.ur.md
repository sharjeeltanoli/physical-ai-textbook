---
sidebar_position: 11
title: Unity for Robotics
description: Leveraging Unity game engine for photorealistic robot simulation
---

# روبوٹکس کے لیے Unity

## تعارف

Unity ایک طاقتور گیم انجن ہے جو روبوٹکس کی سیمولیشن کے لیے استعمال کیا جا سکتا ہے، جو حقیقت پسندانہ گرافکس، موثر رینڈرنگ، اور ایک وسیع ایسٹ ایکو سسٹم فراہم کرتا ہے۔ یہ باب روبوٹکس کی ایپلی کیشنز کے لیے Unity کو ROS 2 کے ساتھ مربوط کرنے کا احاطہ کرتا ہے۔

## روبوٹکس کے لیے Unity کیوں؟

### فائدے

**بصری معیار:**
- حقیقت پسندانہ رینڈرنگ
- جدید روشنی اور سائے
- پوسٹ پروسیسنگ اثرات
- 3D ماڈلز کے ساتھ ایسٹ اسٹور

**کارکردگی:**
- بہتر گیم انجن
- کراس پلیٹ فارم (Windows, Linux, macOS)
- موبائل اور VR/AR سپورٹ

**ترقی:**
- صارف دوست ایڈیٹر
- بصری سکرپٹنگ
- بڑی کمیونٹی

### حدود

- خاص طور پر روبوٹکس کے لیے ڈیزائن نہیں کیا گیا
- فزکس مخصوص سیمولیٹروں کی نسبت کم درست ہوتی ہے
- ROS انٹیگریشن کے لیے سیٹ اپ درکار ہوتا ہے

## Unity روبوٹکس ہب

### آرکیٹیکچر

ROS 2 System                    Unity Simulation
┌─────────────┐                 ┌───────`───────┐
│ ROS 2 Nodes │ ←──────────────→│ Unity Scene  │
│             │   TCP/UDP        │              │
│ Publishers  │   Connections    │ Robot Model  │
│ Subscribers │                  │ Environment  │
└─────────────┘                  └──────────────┘

### اجزاء

1.  **ROS-TCP-Connector** (Unity پیکیج)
2.  **ROS-TCP-Endpoint** (ROS 2 پیکیج)
3.  **URDF Importer** (Unity پیکیج)

## تنصیب

### پیشگی تقاضے

```bash
# Unity Hub (download from unity.com)
# Unity Editor 2021.3 LTS or later
# ROS 2 Humble
```

### ROS-TCP-Endpoint انسٹال کریں

```bash
cd ~/ros2_ws/src
git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### Unity پیکیجز انسٹال کریں

1.  Unity Hub کھولیں → نیا پروجیکٹ بنائیں (3D)
2.  ونڈو → پیکیج مینیجر
3.  گٹ URL سے پیکیج شامل کریں:
    -   `https://github.com/Unity-Technologies/ROS-TCP-Connector.git?path=/com.unity.robotics.ros-tcp-connector`
    -   `https://github.com/Unity-Technologies/URDF-Importer.git?path=/com.unity.robotics.urdf-importer`

## روبوٹ ماڈلز درآمد کرنا

### URDF سے Unity

```csharp
// In Unity Editor:
// Assets → Import Robot from URDF
// Select your robot.urdf file
// Configure import settings:
// - Axis: Y-up
// - Mesh Decomposer: VHACD
// - Import
```

### دستی ترتیب

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

## ROS 2 انٹیگریشن

### Unity سے پبلش کرنا

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

### Unity میں سبسکرائب کرنا

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

## کیمرہ اور سینسر سیٹ اپ

### RGB کیمرہ

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

### ڈیپتھ کیمرہ

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

## ماحول بنانا

### سین سیٹ اپ

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

### لائٹنگ

```csharp
// Directional light (sun)
GameObject sunLight = new GameObject("Sun");
Light light = sunLight.AddComponent<Light>();
light.type = LightType.Directional;
light.intensity = 1.0f;
light.shadows = LightShadows.Soft;
sunLight.transform.rotation = Quaternion.Euler(50, -30, 0);
```

## مصنوعی ڈیٹا کی تیاری

### ڈومین رینڈمائزیشن

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

## لانچ کنفیگریشن

### ROS-TCP-Endpoint شروع کریں

```bash
# In ROS 2 workspace
ros2 run ros_tcp_endpoint default_server_endpoint \
  --ros-args -p ROS_IP:=127.0.0.1
```

### Unity سیٹنگز

```csharp
// In Unity: Robotics → ROS Settings
// ROS IP Address: 127.0.0.1
// ROS Port: 10000
// Protocol: ROS 2
```

## بہترین طریقہ کار

### کارکردگی

-   **Occlusion Culling**: بڑے سینز کے لیے فعال کریں
-   **LOD (تفصیل کی سطح)**: دور دراز اشیاء کے لیے استعمال کریں
-   **بیچنگ**: سٹیٹک میشز کو یکجا کریں

### درستگی

-   **فکسڈ ٹائم اسٹیپ**: مستقل فزکس اپ ڈیٹس سیٹ کریں
    ```csharp
    Time.fixedDeltaTime = 0.02f; // 50Hz
    ```
-   **کولیشن ڈیٹیکشن**: تیز حرکت کرنے والی اشیاء کے لیے Continuous استعمال کریں

### ڈیبگنگ

```csharp
void OnDrawGizmos()
{
    // Visualize sensor ranges
    Gizmos.color = Color.red;
    Gizmos.DrawWireSphere(transform.position, sensorRange);
}
```

## اہم نکات

-   Unity روبوٹکس کے لیے حقیقت پسندانہ بصریات فراہم کرتا ہے
-   ROS-TCP-Connector, ROS 2 انٹیگریشن کو ممکن بناتا ہے
-   URDF امپورٹر روبوٹ ماڈل کی درآمد کو آسان بناتا ہے
-   کیمرے اور سینسر ROS 2 Topics پر پبلش کیے جا سکتے ہیں
-   ڈومین رینڈمائزیشن مضبوط ویژن ماڈلز کو ممکن بناتی ہے
-   کارکردگی کی اصلاح ریئل ٹائم سیمولیشن کے لیے اہم ہے

---

**پچھلا:** [Gazebo سیمولیٹر کے بنیادی اصول](./gazebo-fundamentals)
**اگلا:** [سیمولیشن کے بہترین طریقہ کار](./simulation-best-practices)
