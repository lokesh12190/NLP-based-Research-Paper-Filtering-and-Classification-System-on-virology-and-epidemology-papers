"""
This script filters and classifies academic papers on deep learning applications in virology/epidemiology.
It applies a combined NLP and keyword-based approach to identify relevant papers, classify them by method type, 
and extract specific deep learning methods mentioned. Results are saved in both CSV and Excel formats.
"""

# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define file paths
input_file = 'C:\\Users\\lokes\\Downloads\\nlp\\collection_with_abstracts.csv'
output_file_csv = 'C:\\Users\\lokes\\Downloads\\nlp\\filtered_classified_papers_with_methods_final_code 3.csv'
output_file_excel = 'C:\\Users\\lokes\\Downloads\\nlp\\filtered_classified_papers_with_methods_final_code 3.xlsx'

# Load the dataset
data = pd.read_csv(input_file)

# Define keywords for filtering and classification based on deep learning terms and relevant fields
deep_learning_terms = [
    "neural network", "artificial neural network", "machine learning model", "feedforward neural network",
    "neural net algorithm", "multilayer perceptron", "convolutional neural network", "recurrent neural network",
    "long short-term memory network", "CNN", "GRNN", "RNN", "LSTM", "deep learning", "deep neural networks",
    "computer vision", "vision model", "image processing", "vision algorithms", "computer graphics and vision",
    "object recognition", "scene understanding", "natural language processing", "text mining", "NLP",
    "computational linguistics", "language processing", "text analytics", "textual data analysis", 
    "text data analysis", "text analysis", "speech and language technology", "language modeling", 
    "computational semantics", "generative artificial intelligence", "generative AI", "transformer models", 
    "self-attention models", "transformer architecture", "attention-based neural networks", "transformer networks", 
    "sequence-to-sequence models", "large language model", "LLM", "transformer-based model", 
    "pretrained language model", "generative language model", "foundation model", "multimodal model", 
    "multimodal neural network", "vision transformer", "diffusion model", "continuous diffusion model"
]

fields_of_interest = ["virology", "epidemiology"]

# Keywords grouped by method types for further classification
text_mining_terms = [
    "natural language processing", "text mining", "NLP", "computational linguistics", "language processing", 
    "text analytics", "textual data analysis", "text data analysis", "text analysis", 
    "speech and language technology", "language modeling", "computational semantics"
]

computer_vision_terms = [
    "computer vision", "vision model", "image processing", "vision algorithms", "computer graphics and vision",
    "object recognition", "scene understanding", "vision transformer"
]

# Descriptive text samples for method types used in TF-IDF similarity-based classification
representative_texts = {
    "text mining": "Natural language processing, NLP, text analytics, and computational linguistics.",
    "computer vision": "Image processing, computer vision, and object recognition techniques.",
    "both": "Combination of text mining, NLP, and computer vision techniques.",
    "other": "Methods other than text mining and computer vision."
}

# Vectorize representative texts to compare with paper content for classification
vectorizer = TfidfVectorizer()
method_vectors = vectorizer.fit_transform(representative_texts.values())

# Determine if a paper is relevant by checking for keywords in the title and abstract
def is_relevant_paper(title, abstract):
    text = str(title) + " " + str(abstract)
    text = text.lower()
    has_deep_learning = any(term in text for term in deep_learning_terms)
    has_relevant_field = any(field in text for field in fields_of_interest)
    return has_deep_learning and has_relevant_field

# Classify paper by method type (text mining, computer vision, both, or other) based on keyword presence
def classify_method_type(text):
    text = text.lower()
    has_text_mining = any(term in text for term in text_mining_terms)
    has_computer_vision = any(term in text for term in computer_vision_terms)
    
    if has_text_mining and has_computer_vision:
        return "both"
    elif has_text_mining:
        return "text mining"
    elif has_computer_vision:
        return "computer vision"
    else:
        return "other"

# Extract specific deep learning methods used in each paper, if any
def extract_specific_methods(text):
    text = text.lower()
    found_methods = [method for method in deep_learning_terms if method in text]
    return ", ".join(found_methods) if found_methods else "Not specified"

# Filter, classify, and extract relevant information from the dataset
filtered_data = []
for index, row in data.iterrows():
    title = str(row.get("Title", ""))
    abstract = str(row.get("Abstract", ""))
    combined_text = title + " " + abstract
    
    # Only include papers relevant to deep learning in virology/epidemiology
    if is_relevant_paper(title, abstract):
        method_type = classify_method_type(combined_text)
        specific_method = extract_specific_methods(combined_text)
        filtered_data.append({
            "Title": title,
            "Abstract": abstract,
            "Method Type": method_type,
            "Specific Method": specific_method
        })

# Convert filtered data to DataFrame
filtered_df = pd.DataFrame(filtered_data)

# Save filtered data to both CSV and Excel files for convenience
filtered_df.to_csv(output_file_csv, index=False)
filtered_df.to_excel(output_file_excel, index=False, engine='openpyxl')

# Calculate and print dataset statistics
total_papers_processed = len(data)
total_relevant_papers = len(filtered_data)
method_type_distribution = filtered_df['Method Type'].value_counts().to_dict()

print("Dataset Statistics:")
print(f"Total Papers Processed: {total_papers_processed}")
print(f"Total Relevant Papers: {total_relevant_papers}")
print("Method Types Distribution:", method_type_distribution)

print(f"Filtered and classified data saved to {output_file_csv} and {output_file_excel}")
