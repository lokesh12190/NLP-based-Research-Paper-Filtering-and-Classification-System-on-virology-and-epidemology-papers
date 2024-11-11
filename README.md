
## Table of Contents
- [Project Overview](#project-overview)
- [Solution Components](#solution-components)
- [NLP Techniques for Filtering](#nlp-techniques-for-filtering)
- [Why This Approach is Effective](#why-this-approach-is-effective)
- [Usage Instructions](#usage-instructions)
- [Dataset Statistics](#dataset-statistics)
- [Output Files](#output-files)

## Project Overview

This solution addresses the task of filtering and classifying academic papers that discuss deep learning applications in virology and epidemiology. By leveraging semantic NLP techniques, it goes beyond simple keyword matching, allowing for more accurate and efficient identification of relevant papers. This is particularly useful for researchers needing to quickly locate pertinent studies from large datasets.

## Solution Components

1. **Data Filtering**
   - This component filters papers based on the presence of deep learning and domain-specific terminology within the paper’s title and abstract.
   - The filtering process combines keyword matching with semantic similarity to ensure that selected papers are not only topically relevant but also apply deep learning methodologies in the specified fields.

2. **Method Classification**
   - The system categorizes papers into one of four method types: *text mining*, *computer vision*, *both*, or *other*.
   - A TF-IDF similarity comparison helps to generalize beyond exact keyword matches by comparing paper content with representative text descriptions of each method type.

3. **Specific Method Extraction**
   - After classification, the system identifies and reports the specific deep learning methods discussed in each paper, such as "CNN," "transformer models," or "LSTM."
   - This level of detail provides a nuanced understanding of the methods each paper employs, aiding researchers in pinpointing studies that use particular techniques.

## NLP Techniques for Filtering

This system uses **semantic filtering techniques** rather than simple keyword-based filtering. The approach incorporates:

1. **TF-IDF Vectorization**: The system vectorizes representative text descriptions for each method type to capture their semantic essence. Each paper is then compared with these vectors to assess similarity and categorize the paper accordingly.

2. **Heuristic Rule-Based Filtering**: Keyword lists are combined with heuristic rules to ensure both domain relevance (e.g., virology, epidemiology) and methodological relevance (e.g., deep learning methods) are considered.

These semantic NLP techniques provide flexibility and accuracy by enabling the system to recognize relevant papers even when they do not contain exact keyword matches.

## Why This Approach is Effective

This approach is more effective than traditional keyword-based filtering because:

- **Semantic Generalization**: Using TF-IDF similarity measures allows for a broader and more contextual understanding of paper content, capturing papers that discuss relevant topics even if they use different terminology.
- **High Precision in Targeting**: By focusing on domain-specific terminology and semantic similarity, the system reduces the risk of false positives, ensuring the selected papers are highly relevant.
- **Efficient Processing**: The combination of TF-IDF with rule-based heuristics provides an efficient filtering solution that performs well on large datasets.

## Usage Instructions

### Prerequisites

- **Python 3.8+**
- **Required Python packages**: `pandas`, `scikit-learn`, `openpyxl`
- **File Paths**: Update the paths for `input_file`, `output_file_csv`, and `output_file_excel` as per your local directory.

### Running the Code

1. Ensure that the dataset is located at the path specified in `input_file`.
2. Run the script to process, filter, classify, and save the output.
3. Upon completion, the filtered and classified papers are saved in CSV and Excel formats.

### Output Format

The output files include the following columns:
- **Title**: Paper title
- **Abstract**: Paper abstract
- **Method Type**: Classification as *text mining*, *computer vision*, *both*, or *other*
- **Specific Method**: Specific deep learning methods discussed in the paper (e.g., "CNN," "transformer model").

## Dataset Statistics

- **Total Papers Processed**: 11,450
- **Total Relevant Papers**: 486
- **Method Types Distribution**:
  - Other: 339
  - Text Mining: 126
  - Computer Vision: 20
  - Both: 1

## Output Files

1. **CSV Output**: Filtered and classified data is saved to the specified `output_file_csv`.
2. **Excel Output**: Filtered and classified data is also saved to `output_file_excel` for ease of access.


