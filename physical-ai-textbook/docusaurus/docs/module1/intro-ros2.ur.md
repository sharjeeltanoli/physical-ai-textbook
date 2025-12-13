---
sidebar_position: 4
title: Introduction to ROS 2
description: Understanding the Robot Operating System and its architecture
---

# ROS 2 کا تعارف

## جائزہ

Robot Operating System (ROS) روایتی معنوں میں ایک آپریٹنگ سسٹم نہیں ہے، بلکہ یہ ایک مڈل ویئر فریم ورک ہے جو روبوٹ سافٹ ویئر کے اجزاء کے درمیان ایک منظم مواصلاتی پرت فراہم کرتا ہے۔ ROS 2، جو اس کی دوسری نسل ہے، رئیل ٹائم کارکردگی، سیکیورٹی اور ملٹی-روبوٹ سسٹمز میں نمایاں بہتری لاتا ہے۔

## ROS 2 کیا ہے؟

ROS 2 سافٹ ویئر لائبریریوں اور ٹولز کا ایک سیٹ ہے جو آپ کو روبوٹ ایپلیکیشنز بنانے میں مدد کرتا ہے۔ یہ فراہم کرتا ہے:

- **Hardware abstraction**: سینسرز اور ایکچوایٹرز کے ساتھ یکساں طور پر انٹرفیس
- **Low-level device control**: روبوٹ ہارڈویئر کے ساتھ براہ راست مواصلت
- **Message-passing between processes**: عمل (processes) کے درمیان معیاری مواصلت
- **Package management**: روبوٹ سافٹ ویئر کو منظم کرنا اور شیئر کرنا
- **Visualization and debugging tools**: روبوٹ کے رویے کا معائنہ اور ڈیبگ کرنا

## ROS 1 کے مقابلے میں ROS 2 کیوں؟

### اہم بہتری

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| **Real-time support** | محدود | نیٹیو RTOS سپورٹ |
| **Security** | نہیں | DDS سیکیورٹی پلگ انز |
| **Communication** | کسٹم TCP/UDP | DDS مڈل ویئر |
| **Multi-robot** | مشکل | بلٹ ان سپورٹ |
| **Cross-platform** | بنیادی طور پر Linux | Linux, Windows, macOS |
| **Production ready** | ریسرچ پر توجہ | انڈسٹری کے لیے تیار |

### ROS 2 کب استعمال کریں

✅ **ROS 2 ان کے لیے استعمال کریں:**
- نئے روبوٹ پروجیکٹس
- رئیل ٹائم کنٹرول کے تقاضے
- ملٹی-روبوٹ سسٹمز
- پروڈکشن تعیناتی (Deployments)
- کراس پلیٹ فارم ڈیولپمنٹ

❌ **ROS 1 پر صرف اسی صورت میں غور کریں جب:**
- پرانے سسٹمز کو برقرار رکھنا ہو
- ROS 2 میں مخصوص پیکیج دستیاب نہ ہو

## ROS 2 کا ڈھانچہ

### بنیادی تصورات

┌─────────────────────────────────────────────────┐
│                 Application Layer                │
│  (Your Robot Software - Nodes, Logic, Control)  │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│                  ROS 2 Client Layer              │
│     (rclpy, rclcpp - ROS 2 Client Libraries)    │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│                   ROS 2 Middleware               │
│         (rcl - ROS Client Library Core)         │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│                  DDS Implementation              │
│    (FastDDS, CycloneDDS, RTI Connext, etc.)     │
└─────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────┐
│                   Network Layer                  │
│              (UDP/TCP Transport)                 │
└─────────────────────────────────────────────────┘

### مواصلاتی پیٹرن

ROS 2 متعدد مواصلاتی پیٹرن کو سپورٹ کرتا ہے:

#### 1. Topics (پبلش-سبسکرائب)
```
Publisher Node → Topic → Subscriber Node(s)
   (1-to-many, asynchronous)
```

#### 2. Services (درخواست-جواب)
```
Client Node ↔ Service Node
   (1-to-1, synchronous)
```

#### 3. Actions (ہدف پر مبنی)
```
Action Client → Action Server
   (with feedback and cancellation)
```

#### 4. Parameters
```
Node ↔ Parameter Server
   (configuration management)
```

## DDS: بنیاد

### DDS کیا ہے؟

Data Distribution Service (DDS) ایک مڈل ویئر پروٹوکول ہے جسے ROS 2 مواصلت کے لیے استعمال کرتا ہے۔ یہ فراہم کرتا ہے:

- **Discovery**: خودکار Node کی دریافت (کسی ماسٹر Node کی ضرورت نہیں!)
- **Reliability**: Quality of Service (QoS) پالیسیاں
- **Real-time**: متعینہ مواصلت (Deterministic communication)
- **Scalability**: موثر ملٹی-روبوٹ مواصلت

### QoS پالیسیاں

Quality of Service پالیسیاں مواصلاتی رویے کو کنٹرول کرتی ہیں:

| Policy | Options | Use Case |
|--------|---------|----------|
| **Reliability** | Best effort / Reliable | سینسر ڈیٹا بمقابلہ کمانڈز |
| **Durability** | Volatile / Transient local | رئیل ٹائم بمقابلہ تاریخی |
| **History** | Keep last / Keep all | بفر مینجمنٹ |
| **Deadline** | Duration | رئیل ٹائم رکاوٹیں |

## ROS 2 ڈسٹری بیوشنز

ROS 2 ٹائم بیسڈ ریلیز شیڈول کی پیروی کرتا ہے:

| Distribution | Release Date | EOL | Status |
|--------------|--------------|-----|--------|
| **Foxy Fitzroy** | June 2020 | May 2023 | EOL |
| **Galactic Geochelone** | May 2021 | Nov 2022 | EOL |
| **Humble Hawksbill** | May 2022 | May 2027 | **LTS** ✅ |
| **Iron Irwini** | May 2023 | Nov 2024 | سپورٹڈ |
| **Jazzy Jalisco** | May 2024 | May 2026 | تازہ ترین |

**یہ کورس ROS 2 Humble (LTS) استعمال کرتا ہے**

## انسٹالیشن کی تصدیق

آئیے اپنی ROS 2 انسٹالیشن کی تصدیق کرتے ہیں:

```bash
# Check ROS 2 version
ros2 --version

# List installed packages
ros2 pkg list

# Check environment variables
printenv | grep ROS
```

متوقع آؤٹ پٹ:
```
ros2 cli version: 0.25.x
ROS_VERSION=2
ROS_PYTHON_VERSION=3
ROS_DISTRO=humble
```

## آپ کا پہلا ROS 2 کمانڈ

### چل رہے Nodes کی فہرست

```bash
ros2 node list
```

### ایک ڈیمو Node چلائیں

```bash
# Terminal 1: Run talker
ros2 run demo_nodes_cpp talker

# Terminal 2: Run listener
ros2 run demo_nodes_py listener
```

آپ کو پیغامات پبلش ہوتے اور موصول ہوتے نظر آئیں گے!

### Topics کا معائنہ کریں

```bash
# List all topics
ros2 topic list

# Show message being published
ros2 topic echo /chatter

# Get topic info
ros2 topic info /chatter
```

## ROS 2 ورک اسپیس کا ڈھانچہ

ایک عام ROS 2 ورک اسپیس:

```
ros2_ws/                  # Workspace root
├── src/                  # Source space (your code)
│   ├── package_1/
│   │   ├── package.xml
│   │   ├── CMakeLists.txt (C++)
│   │   ├── setup.py (Python)
│   │   └── ...
│   └── package_2/
├── build/                # Build space (auto-generated)
├── install/              # Install space (executables)
└── log/                  # Build and runtime logs
```

## ROS 2 کے بنیادی تصورات کا خلاصہ

### Nodes
خود مختار processes جو کمپیوٹیشن انجام دیتے ہیں۔ Nodes topics, services، اور actions کے ذریعے مواصلت کرتے ہیں۔

### Packages
ROS 2 کوڈ کے لیے تنظیمی یونٹس۔ متعلقہ nodes, libraries, configs، اور بہت کچھ شامل ہے۔

### Topics
نامزد buses جہاں nodes پیغامات کا تبادلہ کرتے ہیں۔ پبلش-سبسکرائب پیٹرن۔

### Messages
Nodes کے درمیان بھیجی جانے والی ڈیٹا اسٹرکچرز۔ `.msg` فائلوں میں تعریف شدہ۔

### Services
Nodes کے درمیان ہم وقت ساز (synchronous) درخواست-جواب مواصلت۔

### Actions
طویل مدتی کام جن میں فیڈ بیک، نتیجہ، اور ہدف کی منسوخی شامل ہو۔

## ڈیولپمنٹ ٹولز

### کمانڈ لائن انٹرفیس (CLI)

```bash
ros2 <verb> <sub-command> [options]
```

عام verbs:
- `node` - Node کی انتظامیہ
- `topic` - Topic کا معائنہ
- `service` - Service کالز
- `action` - Action کی انتظامیہ
- `param` - Parameter آپریشنز
- `pkg` - Package آپریشنز
- `launch` - Launch فائل کا ایگزیکیوشن

### RQt ٹولز

ROS 2 گرافیکل ٹولز فراہم کرتا ہے:

```bash
# Node graph visualization
rqt_graph

# Topic plotter
rqt_plot

# General purpose GUI
rqt
```

### ڈیبگنگ

```bash
# Console output filtering
ros2 run --prefix 'gdb -ex run --args' <package> <executable>

# Launch file debugging
ros2 launch --debug <package> <launch_file>
```

## شروع سے ہی بہترین طریقے

1. **تفصیلی نام استعمال کریں**: `/robot/sensor/lidar` نہ کہ `/lidar1`
2. **اپنے nodes کو نیم اسپیس کریں**: عالمی نیم اسپیس آلودگی سے بچیں
3. **نام رکھنے کے قواعد پر عمل کریں**: topics کے لیے `snake_case`، nodes کے لیے `CamelCase`
4. **اپنے پیکیجز کی دستاویزات بنائیں**: واضح README فائلیں لکھیں
5. **ورژن کنٹرول استعمال کریں**: پہلے دن سے Git استعمال کریں
6. **بتدریج ٹیسٹ کریں**: بار بار بنائیں اور ٹیسٹ کریں

## عام مسائل (Gotchas)

❗ **Source کرنا بھول جانا**: ہمیشہ بلڈنگ کے بعد `source install/setup.bash` کریں
❗ **Mismatched QoS**: Publisher اور subscriber کے QoS ہم آہنگ ہونے چاہئیں
❗ **Circular dependencies**: پیکیج کی dependencies کو acyclic رکھیں
❗ **Stale builds**: شک کی صورت میں `colcon build --cmake-clean-cache` چلائیں

## اگلے اقدامات

اب جب کہ آپ ROS 2 کی بنیادی باتیں سمجھ چکے ہیں، آئیے اگلے باب میں nodes بنانے اور مواصلاتی پیٹرن ترتیب دینے کا آغاز کریں!

## اہم نکات

- ROS 2 روبوٹ سافٹ ویئر بنانے کے لیے ایک مڈل ویئر فریم ورک ہے
- DDS بنیادی مواصلاتی انفراسٹرکچر فراہم کرتا ہے
- QoS پالیسیاں مواصلت پر باریک کنٹرول دیتی ہیں
- Nodes topics, services، اور actions کے ذریعے مواصلت کرتے ہیں
- ROS 2 Humble (LTS) پروڈکشن کے لیے تیار اور اچھی طرح سے سپورٹڈ ہے

## اضافی وسائل

- [ROS 2 دستاویزات](https://docs.ros.org/en/humble/)
- [ROS 2 ڈیزائن](https://design.ros2.org/)
- [DDS فاؤنڈیشن](https://www.dds-foundation.org/)
- [ROS 2 ٹیوٹوریلز](https://docs.ros.org/en/humble/Tutorials.html)

---

**پچھلا:** [ڈیولپمنٹ انوائرمنٹ سیٹ اپ](../intro/environment-setup)
**اگلا:** [Nodes, Topics، اور Services](./nodes-topics-services)
