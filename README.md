# SemCT-Quality

**Overview:** General model of content quality in the SemCT framework.

**Paper:** "When Diffusion Model Inference Meets MEC Networks: From Independence to Collaboration" --Submitted to IEEE Communications Magazine.

**Experimental Platform:** Ubuntu 20.04 system equipped with an Intel Xeon Gold 6248R CPU and an NVIDIA A100 GPU.


## 1 Environment Setup

Create a new conda environment.

```shell
conda create --name LVM python==3.10
```


## 2 Activate Environment

Activate the created environment.

```shell
source activate LVM
```


## 3 Install Required Packages

ubuntu==20.04  cuda==11.8
```shell
pip install torch==2.4.1
pip install sentence-transformers==3.1.1
pip install diffusers==0.30.3
pip install transformers==4.44.2
pip install accelerate==0.34.2
pip install protobuf==5.28.2
pip install sentencepiece==0.2.0
pip install openai-clip==1.0.1
pip install torchvision==0.19.1
pip install openpyxl==3.1.5
```
Then you should get an env. like:
```shell
Package                  Version
------------------------ --------------------
accelerate               0.34.2
calflops                 0.3.2
certifi                  2025.1.31
charset-normalizer       3.4.1
diffusers                0.30.3
et_xmlfile               2.0.0
filelock                 3.17.0
fsspec                   2025.2.0
ftfy                     6.3.1
fvcore                   0.1.5.post20221221
huggingface-hub          0.28.1
idna                     3.10
importlib_metadata       8.6.1
iopath                   0.1.10
Jinja2                   3.1.5
joblib                   1.4.2
MarkupSafe               3.0.2
mpmath                   1.3.0
networkx                 3.4.2
numpy                    2.2.3
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu12     12.1.0.106
nvidia-nccl-cu12         2.20.5
nvidia-nvjitlink-cu12    12.8.61
nvidia-nvtx-cu12         12.1.105
openai-clip              1.0.1
openpyxl                 3.1.5
packaging                24.2
pillow                   11.1.0
pip                      25.0
portalocker              3.1.1
protobuf                 5.28.2
psutil                   7.0.0
PyYAML                   6.0.2
regex                    2024.11.6
requests                 2.32.3
safetensors              0.5.2
scikit-learn             1.6.1
scipy                    1.15.1
sentence-transformers    3.1.1
sentencepiece            0.2.0
setuptools               75.8.0
sympy                    1.13.3
tabulate                 0.9.0
termcolor                3.0.1
thop                     0.1.1.post2209072238
threadpoolctl            3.5.0
tokenizers               0.19.1
torch                    2.4.1
torchsummary             1.5.1
torchvision              0.19.1
tqdm                     4.67.1
transformers             4.44.2
triton                   3.0.0
typing_extensions        4.12.2
urllib3                  2.3.0
wcwidth                  0.2.13
wheel                    0.45.1
yacs                     0.1.8
zipp                     3.21.0
```


## 4 Locate and Modify StableDiffusion3Pipeline

Open `Demo.py` in your code editor.

Hold down the `ctrl` key if you are on Linux or Windows, or the `command` key if you are on MacOS, and click on StableDiffusion3Pipeline.

![image](/Files/modify.png)

This will navigate to the file `pipeline_stable_diffusion_3.py`.

Replace `pipeline_stable_diffusion.py` with the file of the same name from this repository.


## 5 Explanation of Our Code Files

`Get-Images.py`: Used to generate images with varying shared inference proportion and semantic similarity between personal and public prompts.

`Prompt_Similarity.py`: Used to calculate the semantic similarity matrix.

`CLIP_Calculation.py`: Used to calculate the content quality (i.e., CLIP score).

`Fitting.py`: Used to get the fitting function.


## 6 Generated Content

Our generated image is available in:

Baidu Netdisk: Link: https://pan.baidu.com/s/1GwlOGECB8pqL6Q_FY2vWFw?pwd=4q8x    Extraction code: 4q8x 


## 7 Demo
![image](/Files/SemCT-Github.png)

Public prompt 1: `A graceful cat sitting in a warm and story-rich environment, highlighting its silky fur.`

Public prompt *M*: `A beautifully detailed dog with expressive eyes and a unique coat stands in a scenic natural setting.`

Personal prompt 1: `A fluffy white cat with blue eyes sitting gracefully on a windowsill, bathed in golden sunlight, with a serene garden visible through the window.`

Personal prompt 2: `A gray cat with green eyes, sitting on a wooden porch, with soft sunlight highlighting its fur and a blurred garden in the background.`

Personal prompt *N*: `A majestic dog with striking blue eyes and a muscular build stands alert on a rocky cliff edge, its thick, wavy fur glowing in the golden hour sunlight.` 


## 8 Acknowledge
[Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main): It is a wonderful large vision model.

[CLIP](https://openai.com/index/clip/): It is a neural network that connects text and images.

[Sentence-Transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

[DistributedDiffusion](https://github.com/HongyangDu/DistributedDiffusion): It is the first work on inference sharing in wireless networks.




