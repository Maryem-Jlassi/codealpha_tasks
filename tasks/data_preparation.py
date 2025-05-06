import pandas as pd

def prepare_data(csv1_path, csv2_path, output_path):
    dataset1 = pd.read_csv(csv1_path,sep=',')
    dataset2 = pd.read_csv(csv2_path,sep=',')

    combined_data = pd.concat([dataset1, dataset2], ignore_index=True)

    combined_data = combined_data[['Question', 'Answer']]

    combined_data.to_csv(output_path, index=False)
    print(f"Combined dataset saved to {output_path}")

csv1_path = "data/refugee_questions_answers.csv"
csv2_path = "data/refugee_questions_answers2.csv"
output_path = "data/combined_dataset.csv"

prepare_data(csv1_path, csv2_path, output_path)