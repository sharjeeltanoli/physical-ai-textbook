import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction (Weeks 1-2)',
      items: [
        'intro',
        'intro/course-overview',
        'intro/environment-setup'
      ],
    },
    {
      type: 'category',
      label: 'Module 1: ROS 2 (Weeks 3-5)',
      items: [
        'module1/intro-ros2',
        'module1/nodes-topics-services',
        'module1/first-robot-app',
        'module1/advanced-ros2',
        'module1/best-practices'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Gazebo & Unity (Weeks 6-7)',
      items: [
        'module2/intro-simulation',
        'module2/gazebo-fundamentals',
        'module2/unity-robotics',
        'module2/simulation-best-practices'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac (Weeks 8-10)',
      items: [
        'module3/intro-isaac',
        'module3/isaac-sim-fundamentals',
        'module3/isaac-gym',
        'module3/isaac-ros',
        'module3/advanced-isaac'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: VLA (Weeks 11-13)',
      items: [
        'module4/intro-vla',
        'module4/vla-architecture',
        'module4/training-vla',
        'module4/deploying-vla',
        'module4/vla-applications'
      ],
    },
  ],
};

export default sidebars;
