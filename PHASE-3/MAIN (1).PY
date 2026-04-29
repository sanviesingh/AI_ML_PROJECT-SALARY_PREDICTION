from data_generation import generate_data
from EDA import run_eda
from preprocessing import preprocess_data
from model_training import train_models
from feature_selection import feature_selection_visualization

def main():
    print("Generating Data...")
    df = generate_data()

    print("Performing EDA...")
    run_eda(df)

    print("Running Feature Selection...")
    selected_features = feature_selection_visualization(df)

    print("Preprocessing Data...")
    X, y = preprocess_data(df)

    print("Training Models...")
    train_models(X, y)

if __name__ == "__main__":
    main()