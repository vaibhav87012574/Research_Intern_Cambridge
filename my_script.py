import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm  

def classify_motivations_on_gpu(csv_file_in, csv_file_out):
    # =======================================
    # 🔹 1. LOAD AND PREPARE DATA
    # =======================================
    print("="*80)
    print(f"Reading csv: {csv_file_in}")
    print("="*80)
    
    try:
        df = pd.read_csv(csv_file_in)
        print("...csv loaded successfully..")
    except FileNotFoundError:
        print(f"Error: '{csv_file_in}' not found.")
        print("Please make sure the file is in the same directory.")
        return
    except Exception as e:
        print(f"Error loading csv: {e}")
        return

    # Apply your logic to extract the 'mentor_motivation' column
    df['mentor_motivation'] = df.apply(
        lambda row: row['person1_motivation_mentorship']
        if str(row['person1_is_mentor']).lower() == 'true'
        else row['person2_motivation_mentorship'],
        axis=1
    )

    # Clean the data: ensure it's a list of strings and handle empty/NaN values
    df["mentor_motivation_str"] = df["mentor_motivation"].astype(str).fillna("")
    
    # Filter out empty strings to avoid classifying them
    # We keep the index of these rows to map the results back later
    non_empty_rows = df[df["mentor_motivation_str"] != ""]
    text_to_classify = non_empty_rows["mentor_motivation_str"].tolist()
    
    total_items = len(text_to_classify)
    if total_items == 0:
        print("No mentor motivation text found to classify.")
        # Still, we'll save the file with 'None' for all rows
    else:
        print(f"Found {total_items} non-empty motivation strings to classify.")


    # =======================================
    # 🔹 2. SETUP GPU AND PIPELINE
    # =======================================
    print("="*80)
    print("Loading the zero-shot model...")
    print("="*80)

    # Automatically select GPU (device 0) if available, otherwise CPU (device -1)
    device_to_use = 0 if torch.cuda.is_available() else -1

    if device_to_use == 0:
        print(f"✅ Success! GPU found: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ WARNING: No GPU found. Running on CPU (this will be very slow).")
        print("   If you have an NVIDIA GPU, please check your PyTorch/CUDA install.")

    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device_to_use,
            batch_size=8  # Adjust this (e.g., to 4 or 2) if you get "CUDA out of memory"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    candidate_labels = ["intrinsic motivation", "extrinsic motivation"]
    hypothesis_template = "This text is about {}."

    # =======================================
    # 🔹 3. RUN CLASSIFICATION (WITH PROGRESS BAR)
    # =======================================
    
    # We only run classification if there is text to classify
    if total_items > 0:
        print("="*80)
        print(f"Classifying {total_items} items... 🚀")
        print("="*80)

        try:
            # Create a generator for efficient processing
            text_generator = (text for text in text_to_classify)
            
            # Wrap the classifier call with tqdm to show the progress bar
            results_generator = classifier(
                text_generator,
                candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=False
            )
            
            results = list(tqdm(results_generator, 
                                total=total_items, 
                                desc="Classifying Motivations"))
            
            # Map results back to the original DataFrame using the saved index
            df.loc[non_empty_rows.index, 'motivation'] = [res['labels'][0] for res in results]
            df.loc[non_empty_rows.index, 'confidence'] = [res['scores'][0] for res in results]

            print("\nClassification complete.")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("\n" + "="*80)
                print("ERROR: CUDA out of memory. 💥")
                print("Your 6GB GPU ran out of VRAM. Try a smaller 'batch_size' in the code.")
                print("Change `batch_size=8` to `batch_size=4` or `batch_size=2` and retry.")
                print("="*80)
            else:
                print(f"\nAn unexpected runtime error occurred: {e}")
            return
        except Exception as e:
            print(f"\nError during classification: {e}")
            return
    else:
        print("Skipping classification as no text was found.")

    # =======================================
    # 🔹 4. PROCESS AND SAVE RESULTS
    # =======================================
    
    # Any row that wasn't classified (i.e., was empty) will have NaN.
    # We fill those with "None" as requested.
    df['motivation'] = df['motivation'].fillna("None")
    df['confidence'] = df['confidence'].fillna(0.0)

    print("Final results preview:")
    print(df[["mentor_motivation_str", "motivation", "confidence"]].head(10))

    # Save the final DataFrame
    print("....Saving csv....")
    try:
        df.to_csv(csv_file_out, index=False)
        print(f"✅ Successfully saved results to {csv_file_out}")
    except Exception as e:
        print(f"Error: csv not saved: {e}")

# ==============================================================================
# 🚀 RUN THE SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # Define your input and output filenames
    INPUT_FILE = "merged_conversations_with_translations.csv"
    OUTPUT_FILE = "final_result.csv"
    
    # Run the main function
    classify_motivations_on_gpu(INPUT_FILE, OUTPUT_FILE)