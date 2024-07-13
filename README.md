# Data Compression Challenge Evaluation

This repository contains the evaluation code for the Data Compression Challenge. The code evaluates the performance of submitted models on CIFAR-100 and Tiny ImageNet datasets. The evaluation is performed on GPU if available.

## Requirements

- Python==3.10
- PyTorch==2.3.0

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/data-compression-challenge-evaluation.git
   cd data-compression-challenge-evaluation
2. Install the required packages:

   ```bash
   pip install torch numpy


## Usage

1. Ensure your submission file and the reference data are organized as follows:

```csharp
input/
├── res/
│   ├── cifar100.pt
│   └── tinyimagenet.pt
└── ref/
    ├── cifar100_test.pt
    └── tinyimagenet_test.pt
```

2. Run the evaluation script:

```bash
python evaluate.py input output
```
- `input` is the directory containing the res and ref subdirectories.
- `output` is the directory where the evaluation scores will be saved.

- `res` is the file you need to complete the contest submission, see the contest website for details：https://codalab.lisn.upsaclay.fr/competitions/19577 .

- `ref` is the test dataset, the download link is : https://pan.baidu.com/s/1BZ1omnzDeIj0XGM8kaXqlQ?pwd=8atr ，and the Extract code: 8atr .

## Script Explanation
- `evaluate.py`:
  
    - Loads the distilled train data from the submission file.
    - Normalizes the data if necessary.
    - Loads the test data and labels from the reference files.
    - Defines a simple Convolutional Neural Network (CNN) for classification.
    - Trains the CNN on the distilled data.
    - Evaluates the trained model on the test data.
    - Computes and outputs the average accuracy over three runs.
 
## Output
The evaluation script generates a file named scores.txt in the output directory containing the accuracy score of the submitted models.

## Troubleshooting
- Ensure that the input directory structure matches the expected format.
- Verify that the .pt files contain the expected data and are not corrupted.
- Make sure you have a compatible version of PyTorch installed.
