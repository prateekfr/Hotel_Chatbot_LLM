import os
import json
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent

hotel_urls = [
    "https://www.tajhotels.com/en-in/hotels/taj-mahal-palace-mumbai",
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
os.makedirs(RAW_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def get_driver():
    options = Options()
    # options.add_argument("--headless=new")
    options.add_argument(f"user-agent={UserAgent().random}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=options)

def scroll_page(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(6):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

def scrape_hotel(url):
    driver = get_driver()
    data = {
        "url": url,
        "hotel_name": None,
        "address": None,
        "check_in": None,
        "check_out": None,
        "rooms_and_suites": None,
        "contact": {}
    }
    try:
        # Main hotel info
        driver.get(url)
        time.sleep(2)
        scroll_page(driver)
        try:
            name_elem = driver.find_element(By.CSS_SELECTOR, ".MuiTypography-root.MuiTypography-heading-l.css-gw9im6")
            data["hotel_name"] = name_elem.text.strip()
        except Exception as e:
            logging.warning(f"Hotel name not found: {e}")

        try:
            addr_elem = driver.find_element(By.CSS_SELECTOR, ".MuiTypography-root.MuiTypography-body1.css-11bosi7")
            data["address"] = addr_elem.text.strip()
        except Exception as e:
            logging.warning(f"Address not found: {e}")

        try:
            times = driver.find_elements(By.CSS_SELECTOR, ".MuiTypography-body1")
            for t in times:
                txt = t.text.strip().lower()
                if "check-in" in txt:
                    data["check_in"] = t.text.strip()
                elif "check-out" in txt:
                    data["check_out"] = t.text.strip()
        except Exception as e:
            logging.warning(f"Check-in/Check-out not found: {e}")

        try:
            for el in driver.find_elements(By.CSS_SELECTOR, ".MuiTypography-body1"):
                txt = el.text.strip().lower()
                if "room" in txt and "suite" in txt:
                    data["rooms_and_suites"] = el.text.strip()
                    break
        except Exception as e:
            logging.warning(f"Rooms & Suites not found: {e}")

        try:
            facilities = {}
            category_blocks = driver.find_elements(By.CSS_SELECTOR, "div.MuiGrid-item")

            for category in category_blocks:
                try:
                    # Click "...more" if available to expand the section
                    try:
                        more_button = category.find_element(By.XPATH, './/span[contains(text(), "...more")]')
                        if more_button.is_displayed():
                            driver.execute_script("arguments[0].click();", more_button)
                            WebDriverWait(category, 2).until(
                                EC.invisibility_of_element(more_button)
                            )
                    except Exception as e:
                        # No "...more" present or error clicking it — continue silently
                        pass

                    category_title_elem = category.find_element(By.CSS_SELECTOR, "h4.MuiTypography-heading-xs")
                    category_title = category_title_elem.text.strip()

                    features = []
                    feature_elements = category.find_elements(By.CSS_SELECTOR, "span.MuiTypography-body-ml")
                    for feature in feature_elements:
                        feature_text = feature.text.strip()
                        if feature_text:
                            features.append(feature_text)

                    facilities[category_title] = features
                except Exception as e:
                    logging.warning(f"Failed to extract one facility block: {e}")



            for category, items in facilities.items():
                    
                data[category] = items
        except Exception as e:
            logging.warning(f"Facilities section not found: {e}")
        if "HOTEL" in data:
            data["amenities"] = data.pop("HOTEL")
            data["dining_options"] = data.pop("DINING")
            data["wellness_and_spa"] = data.pop("WELLNESS")
            data["room_info"] = data.pop("ROOMS")
            data.pop("GETTING HERE")
            data.pop("OUR BRANDS")
        contact = {}    
        try:
            body_text = driver.find_element(By.TAG_NAME, "body").text
            import re
            phones = re.findall(r"(\+?\d[\d\s\-]{8,}\d)", body_text)
            emails = re.findall(r"[\w\.-]+@[\w\.-]+", body_text)
            if phones:
                contact["phone"] = phones[0]
            if emails:
                contact["email"] = emails[0]
            data["contact"] = contact
        except:
            data["contact"] = {}

    finally:
        driver.quit()

    from bs4 import BeautifulSoup
    import requests
    url="https://www.tajhotels.com/en-in/hotels/taj-mahal-palace-mumbai/places-to-visit"

    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, 'html.parser')
    attraction_names = []
    for tag in soup.find_all(['h2']):
        text = tag.get_text(strip=True)
        if text and len(text.split()) <= 6 and not text.isupper():
            attraction_names.append(text)
    unique_names = list(set(attraction_names))
    data["local_attractions"] = unique_names

    return data

def main():
    for url in hotel_urls:
        logging.info(f"Scraping: {url}")
        info = scrape_hotel(url)
        outname = url.rstrip("/").split("/")[-1]
        out_path = os.path.join(RAW_DIR, f"{outname}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
