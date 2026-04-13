import os

print("--- STARTING FOLDER DETECTIVE ---")
print(f"Current Directory: {os.getcwd()}\n")

found_cities = []
for root, dirs, files in os.walk('.'):
    # If a folder contains the change mask, it's a city folder!
    if 'cm.png' in files or 'cm' in dirs:
        found_cities.append(root)

if len(found_cities) > 0:
    print(f"SUCCESS! Found your cities hiding in these folders:")
    for city in found_cities[:5]: # Print first 5 so it doesn't spam you
        print(f" -> {city}")
    print(f"...and {len(found_cities) - 5} more.")
else:
    print("ERROR: Could not find any folder containing 'cm' or 'cm.png' anywhere!")