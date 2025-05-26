# Project Title
# Billiard Robot Control and Simulation System

# Description
This project is a comprehensive system for controlling a billiard-playing robot. It includes modules for computer vision-based ball detection, physics simulation for predicting shot outcomes, a graphical user interface for visualization, and communication interfaces for robot control.

# Project Structure
The project is organized into several main directories:

*   `main/`: Contains the core source code of the application.
    *   `core/`: Includes the main logic for ball generation, shot simulation, and the billiard API.
    *   `vision/`: Houses modules for image capture, ball detection (using techniques like Hough transforms and YOLO), and camera calibration.
    *   `communicate/`: Manages TCP communication for controlling the robot or other external interfaces.
    *   `gui/`: Provides the graphical user interface for visualizing simulations and system status.
    *   `cli/`: Contains the command-line interface for interacting with the system, for example, to execute shots.
    *   `configs/`: Stores configuration files and settings for the application.
    *   `captured_images/`: Default directory for storing images captured by the vision system.
    *   `captures_json/`: Default directory for storing JSON data related to captures.
*   `tools/`: Includes various utility scripts for development, testing, or specific operations like camera calibration or data conversion.
*   `captures_json/`: Seems to be a top-level directory for storing JSON data, potentially outputs or logs.

Key files in the root directory:

*   `README.md`: This file.
*   `CORDS.json`: Likely contains coordinate data or calibration information.
*   `best.pt`: A PyTorch model file, probably a trained YOLO model for object detection.
*   `handeye_result.yaml`: Stores the results of hand-eye calibration between the robot arm and the camera.
*   `package.json`: Indicates Node.js dependencies, possibly for a part of the GUI or a tool.

# Setup and Execution
This section outlines the general steps to set up and run the project.

**Prerequisites:**

*   **Python:** The core of the project is written in Python. Ensure you have a recent version of Python 3 installed (e.g., Python 3.8+).
*   **pip:** Python's package installer, used to install project dependencies.
*   **Node.js and npm:** The presence of a `package.json` file suggests that Node.js and npm might be required for parts of the GUI or certain tools. Install a recent LTS version.
*   **Git:** For cloning the repository.

**General Setup Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install Python dependencies:**
    A `requirements.txt` file is commonly used for Python projects. If one exists, run:
    ```bash
    pip install -r requirements.txt
    ```
    If not, you may need to identify and install dependencies manually based on import errors (e.g., `opencv-python`, `numpy`, `torch`, `torchvision`, `websockets`, `PyQt5` or `tkinter` for GUI).
    *(Developer Note: It is highly recommended to create and maintain a `requirements.txt` file for this project.)*

3.  **Install Node.js dependencies (if applicable):**
    If you intend to use parts of the project that rely on Node.js (e.g., a web-based GUI or specific tools):
    ```bash
    npm install
    ```

4.  **Camera Setup and Calibration:**
    *   Ensure your camera is connected and configured correctly.
    *   Run any necessary camera calibration scripts (potentially found in `tools/` or `main/vision/`). This may involve generating `intrinsics.yaml` and `handeye_result.yaml`.

5.  **Configuration:**
    *   Review and adjust settings in `main/configs/setting.py` or other configuration files as needed.
    *   Ensure files like `CORDS.json` are correctly populated if they are crucial for operation.

**Running the System:**

*   **Main Application:** The primary entry point is likely `main/main.py`.
    ```bash
    python main/main.py
    ```
*   **Command-Line Interface:** The CLI can be accessed via `main/cli/shot_cli.py`.
    ```bash
    python main/cli/shot_cli.py --help
    ```
*   **Specific Tools:** Scripts in the `tools/` directory can be run directly. For example:
    ```bash
    python tools/some_tool.py
    ```

*(Note: This is a general guide. Specific components might have their own setup instructions or dependencies. Refer to individual module documentation or code comments if available.)*

# Functionalities
This project integrates several key functionalities to achieve automated billiard play:

*   **Ball Detection and Localization:**
    *   Utilizes computer vision techniques to detect billiard balls on the table from a camera feed.
    *   Supports multiple detection methods, including Hough Circle Transform and YOLO (You Only Look Once) deep learning models (`best.pt`).
    *   Calibrates camera intrinsic and extrinsic parameters (`intrinsics.yaml`, `handeye_result.yaml`) to accurately map ball positions from image coordinates to real-world (table) coordinates.
    *   Stores detected ball coordinates, possibly in `CORDS.json` or within `captures_json/`.

*   **Shot Calculation and Simulation:**
    *   The `main/core/solver_core.py` likely contains the physics engine to simulate ball trajectories, collisions, and pocketing.
    *   `main/core/billiard_api.py` probably provides an API to this solver, allowing other modules to request shot solutions.
    *   `main/core/ball_generator.py` might be used for setting up specific ball layouts for simulation or testing.

*   **Robot Control Interface:**
    *   The `main/communicate/` modules (e.g., `tcp_communicate.py`) suggest the capability to send commands to a robot arm or cueing mechanism via TCP/IP.
    *   This allows the system to execute the calculated optimal shot.

*   **Graphical User Interface (GUI):**
    *   `main/gui/simulator.py` and `main/gui/visualize.py` provide tools to visualize the detected balls, the table state, and simulated shot trajectories.
    *   This is crucial for monitoring, debugging, and potentially for interactive use.

*   **Command-Line Interface (CLI):**
    *   `main/cli/shot_cli.py` offers a way to interact with the system from the command line, perhaps for triggering specific actions, running tests, or controlling shots without the full GUI.

*   **Image and Data Management:**
    *   The system captures and stores images (e.g., in `main/captured_images/`).
    *   It also manages JSON data related to captures and coordinates (e.g., in `main/captures_json/` and the root `captures_json/`).

# Tools and Utilities
The `tools/` directory contains various scripts and utilities that can be helpful for development, testing, calibration, or data processing related to this project. Refer to the specific scripts within that directory for their individual functionalities.

# Contributing
Contributions to this project are welcome! If you'd like to contribute, please follow these general guidelines:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
    or
    ```bash
    git checkout -b fix/your-bug-fix-name
    ```
3.  **Make your changes.** Ensure your code adheres to any existing style guidelines and include tests if applicable.
4.  **Commit your changes** with a clear and descriptive commit message.
5.  **Push your branch** to your fork:
    ```bash
    git push origin feature/your-feature-name
    ```
6.  **Open a pull request** against the main repository.

Please ensure your pull request describes the problem and solution, and any other relevant information.

# License
This project is licensed under the [NAME OF LICENSE] - see the LICENSE.md file for details (if one exists).

If no LICENSE.md file is present, please assume the code is proprietary or contact the project maintainers for licensing information.
