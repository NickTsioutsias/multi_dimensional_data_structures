"""
Test script to load and explore the coffee data
"""
from src.utils.data_loader import CoffeeDataLoader

# Load the simplified dataset
print("Loading simplified coffee dataset...")
loader = CoffeeDataLoader('data/simplified_coffee.csv')

# Load and preprocess the data
df = loader.preprocess()

# Get the 4D points
points = loader.get_points()

print(f"\nData shape: {points.shape}")
print(f"This means we have {points.shape[0]} coffee reviews")
print(f"Each with {points.shape[1]} dimensions")

# Show a sample data point
print("\nFirst data point:")
print(f"  Year: {points[0][0]:.0f}")
print(f"  Rating: {points[0][1]:.1f}")
print(f"  Price: ${points[0][2]:.2f}")
print(f"  Country ID: {points[0][3]:.0f}")

# Show the actual coffee info
first_coffee = df.iloc[0]
print(f"\nThis corresponds to:")
print(f"  Coffee: {first_coffee['name']}")
print(f"  Roaster: {first_coffee['roaster']}")
print(f"  Country: {first_coffee['loc_country']}")