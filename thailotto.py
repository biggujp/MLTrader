import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

# =========================
# 🔄 Source 1 (lottery.co.th)
# =========================
def fetch_source_1(pages=10):
    results = []

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for page in range(1, pages + 1):
        url = f"https://www.lottery.co.th/small?page={page}"
        try:
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            rows = soup.select("table tbody tr")

            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    date = cols[0].text.strip()
                    num = cols[2].text.strip()

                    if num.isdigit():
                        results.append((date, num.zfill(2)))

            print(f"Source1 page {page} OK")

        except Exception as e:
            print("Source1 error:", e)

        time.sleep(1)

    return results


# =========================
# 🔄 Source 2 (huay.com)
# =========================
def fetch_source_2(pages=5):
    results = []

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for page in range(1, pages + 1):
        url = f"https://huay.com/lottery/result?page={page}"

        try:
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            rows = soup.select("table tbody tr")

            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 3:
                    date = cols[0].text.strip()
                    num = cols[2].text.strip()

                    if num.isdigit():
                        results.append((date, num.zfill(2)))

            print(f"Source2 page {page} OK")

        except Exception as e:
            print("Source2 error:", e)

        time.sleep(1)

    return results


# =========================
# 🧹 Clean + Merge
# =========================
def clean_and_merge(data):
    df = pd.DataFrame(data, columns=["date", "number"])

    # ลบค่าแปลก
    df = df[df["number"].str.isdigit()]

    # format
    df["number"] = df["number"].astype(str).str.zfill(2)

    # remove duplicates
    df = df.drop_duplicates()

    # sort by date (optional)
    df = df.sort_values(by="date")

    return df


# =========================
# 💾 Save CSV
# =========================
def save_csv(df, filename="thai_lotto_2digit.csv"):
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} rows to {filename}")


# =========================
# 🚀 MAIN
# =========================
if __name__ == "__main__":
    print("🔄 Fetching data...")

    data1 = fetch_source_1(pages=10)
    data2 = fetch_source_2(pages=5)

    all_data = data1 + data2

    print(f"Total raw records: {len(all_data)}")

    df = clean_and_merge(all_data)

    save_csv(df)

    print("\nPreview:")
    print(df.head())