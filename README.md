# Frog-Calls-Recognition
A combined fine-tuned model built on top of Perch (Google) pre-trained model for frog calls detection of 16 different frog species in Victoria, Australia. For live testing and model training, please refer to the Colab' notebooks, for local deployment, please refer to the Python script.

## 1. For local deployment (Linux)
**Note:** This demo was developed using Python 3.10.16. It requires Python version > 3.10 and < 3.12 for compatibility.
### Step 1. Clone this repository to your workspace
```bash
git clone https://github.com/thieulong/Frog-Calls-Recognition.git
```
### Step 2. Create and activate a virtual environment for the dependencies in this project
```bash
python -m venv fcr
source fcr/bin/activate
```
With `fcr` is the name of the virtual environment, you can change this base on your preferrence, since the installation will be complicated hence it's best if we do it in a virtual environment for ease of management.
### Step 3. Install external dependencies
- **perch**  
   To install and configure **perch**, please follow the instructions provided in the official [perch repo](https://github.com/google-research/perch), under the **Installation** section.
    
- **perch-hoplite**  
   This is an addition dependency of **perch** that we also need to install along, please also follow the instructions provided in the [perch-hoplite repo](https://github.com/google-research/perch-hoplite/tree/main), under the **Installation** section

- **chirp**  
  After installing Perch, you’ll see a folder named `perch` in your current workspace. Open that folder, locate the `chirp` directory inside it, and move (or drag) it out into your main workspace directory.

After you have done the above installations, assume you're still in the virtual environment, please install the relevant Python dependencies with  
```bash
pip install -r requirements.txt
```
### Step 4. Run the main script
