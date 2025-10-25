# The scaling law of multiple long-term condition accumulation over the life course

This repository houses the scripts utilised in mean cumulative count calculation and power-law model fitting as detailed in our study.

## Running Environment

The Python scripts in this repository are designed to run within a Python environment. Key requirements include:

- Python Version: 3.8.18.
- RAM Configuration: This code was developed and tested on a workstation with 256GB system RAM.

## Directory Structure

```
./
├── Run_Experiment.ipynb        # Main analysis notebook 
├── README.md                   # This file
├── data/                       # Input data directory
│   └── df_patient_20_80.csv    # Main patient dataset
└── results/                    # Output directory
```

## Dependencies

The script requires the following Python packages:
- `numpy`
- `pandas`
- `tqdm`

## Input Requirements

The input CSV file must contain the following columns:
## Dataset Column Descriptions

### Patient Demographic Information
- <span style="color:red">**patid**</span>: Unique patient identifier
- <span style="color:red">**gender**</span>: Biological sex (M = Male, F = Female)
- <span style="color:red">**yob**</span>: Year of birth

### Patient Follow-up Timeline
- <span style="color:red">**exit_time**</span>: Date when patient was lost to follow-up. For patients followed beyond December 31, 2019, this value is set to 31/12/2019
- <span style="color:red">**exit_age**</span>: Age at exit, calculated as `exit_time.year - yob`
- <span style="color:red">**die_age**</span>: Age at death (if applicable), derived from death records

### Medical Conditions
- <span style="color:red">**Condition Flags**</span>: 
  - `'Anx', 'Dep', 'SMI', 'Ast', 'COPD', 'Diab', 'Hyp', 'CHD', 'StroTIA', 'AF', 'HF', 'PAD', 'CKD', 'Dem', 'Park', 'Ost', 'RA', 'Can'`
  - Contains onset date/year for each condition
  - `NaN` indicates the condition was not diagnosed
  - Assessment period: through end of 2019 or until patient exit
  - The 18 conditions are:
Stroke/TIA; 
Coronary heart disease (CHD); 
Atrial Fibrillation (AF); 
Heart Failure (HF); 
Hypertension; 
Peripheral Arterial Disease (PAD); 
Diabetes Mellitus; 
Asthma; 
Chronic Obstructive Pulmonary Disease (COPD); 
Dementia; 
Parkinson’s Disease; 
Depression; 
Anxiety; 
Serious Mental Illnesses (Bipolar Disorder and Schizophrenia); 
Cancer excluding non-melanoma skin cancers; 
Chronic Kidney Disease (CKD); 
Osteoporosis; 
Rheumatoid Arthritis (RA).

### Condition Summary Metrics
- <span style="color:blue">**n_cond**</span>: Total number of conditions diagnosed by end of 2019 or patient exit
- <span style="color:blue">**1st_cond**</span>: First diagnosed condition (index condition), `Health` means no condition by end of 2019.
- <span style="color:blue">**1st_age**</span>: Age at first condition diagnosis
- <span style="color:blue">**1st_time**</span>: Date of first condition diagnosis

---

### Color Coding Legend:
- <span style="color:red">Red</span>: Core patient data and timeline variables
- <span style="color:blue">Blue</span>: Derived summary metrics


## Analysis Details

### Age Bands
The analysis uses predefined age bands:
`[[20,29], [30,39], [40,49], [50,59], [60,69], [70,79]]`

### Medical Conditions
The analysis includes 18 medical conditions:
`['Anx','Dep','SMI','Ast','COPD','Diab','Hyp','CHD','StroTIA','AF','HF','PAD','CKD','Dem','Park','Ost','RA','Can']`


