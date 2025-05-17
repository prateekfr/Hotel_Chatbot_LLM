import json
import os

INPUT_FILE = "data/raw/taj-mahal-palace-mumbai.json"
OUTPUT_FILE = "data/processed/processed_dataset.json"

def generate_qa_pairs(hotel_data):
    qa_pairs = []
    name = hotel_data["hotel_name"]

    # Basic Info
    qa_pairs.extend([
        {"question": f"What is the name of the hotel located at {hotel_data['address']}?", "answer": name},
        {"question": f"What is the address of {name}?", "answer": hotel_data["address"]},
        {"question": f"When can I check in and out at {name}?", "answer": f"{hotel_data['check_in']} and {hotel_data['check_out']}."},
        {"question": f"How many rooms and suites are available at {name}?", "answer": hotel_data["rooms_and_suites"]},
    ])

    # Contact
    contact = hotel_data.get("contact", {})
    qa_pairs.append({
        "question": f"How can I contact {name}?",
        "answer": f"Phone: {contact.get('phone', 'N/A')} | Email: {contact.get('email', 'N/A')}"
    })

    # Amenities
    for amenity in hotel_data.get("amenities", []):
        qa_pairs.append({
            "question": f"Is {amenity.lower()} available at {name}?",
            "answer": f"Yes, {name} offers {amenity}."
        })

    qa_pairs.append({
        "question": f"What amenities does {name} offer?",
        "answer": "; ".join(hotel_data["amenities"])
    })

    # Dining Options
    for dining in hotel_data.get("dining_options", []):
        qa_pairs.append({
            "question": f"What kind of dining option is {dining.split('(')[0].strip()} at {name}?",
            "answer": dining
        })

    qa_pairs.append({
        "question": f"What are the dining options at {name}?",
        "answer": "; ".join(hotel_data["dining_options"])
    })

    # Wellness & Spa
    for facility in hotel_data.get("wellness_and_spa", []):
        qa_pairs.append({
            "question": f"Is {facility.lower()} available at {name}?",
            "answer": f"Yes, {name} provides {facility}."
        })

    qa_pairs.append({
        "question": f"What wellness and spa facilities are available at {name}?",
        "answer": "; ".join(hotel_data["wellness_and_spa"])
    })

    # Room Info
    for feature in hotel_data.get("room_info", []):
        qa_pairs.append({
            "question": f"Do the rooms at {name} offer {feature.lower()}?",
            "answer": f"Yes, rooms may offer {feature}."
        })

    qa_pairs.append({
        "question": f"What room features are available at {name}?",
        "answer": "; ".join(hotel_data["room_info"])
    })

    # Attractions
    for place in hotel_data.get("local_attractions", []):
        qa_pairs.append({
            "question": f"Is {place} close to {name}?",
            "answer": f"Yes, {place} is a nearby attraction to {name}."
        })

    qa_pairs.append({
        "question": f"What are the nearby attractions to {name}?",
        "answer": "; ".join(hotel_data["local_attractions"])
    })

    return qa_pairs


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        hotel_data = json.load(f)

    qa_pairs = generate_qa_pairs(hotel_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(qa_pairs)} Q&A pairs → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
