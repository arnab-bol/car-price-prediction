
import joblib
import pandas as pd

model = joblib.load("models/best_model.joblib")

print("\n===== CAR SELLING PRICE PREDICTION =====")
print("Enter the car details below.\n")

try:
    present_price = float(input("Showroom price (in dataset units): "))
    kms_driven = float(input("Kilometers driven: "))
    fuel_type = input("Fuel type (Petrol/Diesel/CNG): ").strip().title()
    seller_type = input("Seller type (Dealer/Individual): ").strip().title()
    transmission = input("Transmission (Manual/Automatic): ").strip().title()
    owner = int(input("Number of previous owners (0, 1, 2...): "))
    car_age = float(input("Car age in years: "))

    if present_price <= 0 or kms_driven < 0 or car_age < 0 or owner < 0:
        raise ValueError("Prices must be positive; other values cannot be negative.")

    if fuel_type not in ["Petrol", "Diesel", "Cng"]:
        raise ValueError("Fuel type must be Petrol, Diesel, or CNG.")

    if seller_type not in ["Dealer", "Individual"]:
        raise ValueError("Seller type must be Dealer or Individual.")

    if transmission not in ["Manual", "Automatic"]:
        raise ValueError("Transmission must be Manual or Automatic.")

    car = pd.DataFrame([{
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner,
        "Car_Age": car_age
    }])

    prediction = model.predict(car)[0]

    print("\n" + "=" * 42)
    print("         CAR PRICE PREDICTION")
    print("=" * 42)
    print(f"Estimated selling price: {prediction:.2f}")
    print("Units are the same as Selling_Price in car.csv.")
    print("This is an estimate, not a guaranteed market price.")
    print("=" * 42)

except ValueError as error:
    print("Invalid input:", error)
