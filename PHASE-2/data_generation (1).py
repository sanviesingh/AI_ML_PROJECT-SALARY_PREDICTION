import pandas as pd
import numpy as np

def generate_data(n=500):
    np.random.seed(42)

    data = pd.DataFrame({
        "age": np.random.randint(21, 60, n),
        "experience": np.random.randint(0, 35, n),
        "education": np.random.choice(["Bachelors", "Masters", "PhD"], n),
        "job_level": np.random.choice(["Junior", "Mid", "Senior"], n),
        "department": np.random.choice(["IT", "HR", "Finance", "Sales"], n),
        "job_role": np.random.choice(["Developer", "Manager", "Analyst"], n),
        "performance": np.random.randint(1, 6, n),
        "projects": np.random.randint(1, 20, n),
        "certifications": np.random.randint(0, 10, n),
        "weekly_hours": np.random.randint(30, 60, n),
        "overtime": np.random.randint(0, 20, n),
        "team_size": np.random.randint(1, 15, n),
        "leadership_score": np.random.rand(n) * 10,
        "skill_score": np.random.rand(n) * 10,
        "years_current_role": np.random.randint(0, 10, n),
        "promotion_count": np.random.randint(0, 5, n),
        "employment_type": np.random.choice(["Full-time", "Part-time"], n),
        "location": np.random.choice(["Urban", "Rural"], n),
        "company_size": np.random.choice(["Small", "Medium", "Large"], n),
        "job_satisfaction": np.random.randint(1, 6, n)
    })

    data["salary"] = (
        20000 +
        data["experience"] * 1500 +
        data["performance"] * 2000 +
        data["projects"] * 300 +
        data["skill_score"] * 1000 +
        np.random.randint(-5000, 5000, n)
    )

    return data

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("employee_data.csv", index=False)
    print("Dataset created!")