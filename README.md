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
### Step 4. Download the models
As a temprorary solution, for now please download the `Frog-Models` on this [shared Drive](https://drive.google.com/drive/folders/1a6mqlnyKtJkcQTo9i0ZJv0ScNeVzC6oJ?usp=sharing). And store in this repo directory, the overall file structure after this step should be as below
```bash
├── chirp
│   ├── audio_utils.py
│   ├── birb_sep_paper
│   ├── configs
│   ├── config_utils.py
│   ├── crawl
│   ├── data
│   ├── eval
│   ├── export_utils.py
│   ├── inference
│   ├── __init__.py
│   ├── models
│   ├── path_utils.py
│   ├── preprocessing
│   ├── projects
│   ├── __pycache__
│   ├── signal.py
│   ├── tests
│   ├── train
│   └── train_tests
├── config.json
├── fcr_env
│   ├── bin
│   ├── etc
│   ├── include
│   ├── lib
│   ├── lib64 -> lib
│   ├── local
│   ├── pyvenv.cfg
│   ├── share
│   └── src
├── Frog-Models
│   ├── Crinia parinsignifera
│   ├── Crinia signifera
│   ├── embeddings
│   ├── Geocrinia laevis
│   ├── Geocrinia victoriana
│   ├── Limnodynastes dumerilii
│   ├── Limnodynastes peronii
│   ├── Limnodynastes tasmaniensis
│   ├── Litoria ewingii
│   ├── Litoria fallax
│   ├── Litoria lesueuri
│   ├── Litoria peronii
│   ├── Litoria raniformis
│   ├── Litoria verreauxii
│   ├── Neobatrachus sudelli
│   ├── Pseudophryne bibroni
│   └── Pseudophryne semimarmorata
├── Input
│   ├── 1.WAV
│   ├── 2.WAV
│   ├── 3.WAV
│   └── 4.WAV
├── main.py
├── Output
│   └── embeddings
├── perch
│   ├── agile_modeling.ipynb
│   ├── analysis.ipynb
│   ├── AUTHORS
│   ├── CONTRIBUTING.md
│   ├── Dockerfile
│   ├── embed_audio.ipynb
│   ├── LICENSE
│   ├── poetry.lock
│   ├── pyproject.toml
│   └── README.md
├── README.md
├── requirements.txt
└── results-test.csv
```
### Step 5. Run the main script
```bash
python main.py
```
Run the `main.py` script to execute the model on a set of test audio files located in the `Input/` directory. If the script runs successfully, it will generate several .csv files in the `Output/` folder. The main recognition results can be found in `results-all.csv`.  

Among the test files, only `1.WAV` and `2.WAV` contain frog calls — the rest are background noise. Once `results-all.csv` is generated, you can compare it with `results-test.csv` provided in this repository. If the results match, it confirms that the model has been successfully loaded and executed.
### Step 6. Modify parameters
This program works by putting your raw audio files (.wav/.mp3) in the `Input/` folder, run the `main.py` and it will generate detection results in the `Output/` folder, in .csv files. You can modify the parameters by looking at the `config.json` file in this repository, the content should be as below
```bash
{
  "input_dir": "Input",
  "output_dir": "Output",
  "model_dir": "Frog-Models",
  "working_dir": "Output",
  "model_choice": "perch_8",
  "default_threshold": 2.5,
  "detect_species": "All species"
}
```
In which,
- `input_dir`: where you will store all of the raw audio data
- `ouput_dir`: where the results will be generated
- `model_dir`: the model directories you downloaded from Drive
- `working_dir`: where some temporary files will be generated during the running the model (embeddings), you can point this to `output_dir`
- `model_choice`: leave this as perch_8 as it's the best pre-trained model for this application
- `default_threshold`: the minimum confidence score to count as detection, 2.5 is corresponding to > 92%
- `detect_species`: Choose the species you want to detect, "All species" will detect all trained species, you can choose specific species by changing it to the name of that species, such as "Litoria raniformis", there are 16 species, listed as below:
    - Crinia parinsignifera
    - Crinia signifera
    - Geocrinia laevis
    - Geocrinia victoriana
    - Limnodynastes dumerilii
    - Limnodynastes peronii
    - Limnodynastes tasmaniensis
    - Litoria ewingii
    - Litoria fallax
    - Litoria lesueuri
    - Litoria peronii
    - Litoria raniformis
    - Litoria verreauxii
    - Neobatrachus sudelli
    - Pseudophryne bibroni
    - Pseudophryne semimarmorata
