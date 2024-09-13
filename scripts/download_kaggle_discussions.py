import requests
from bs4 import BeautifulSoup
import json
import os
import time
from urllib.parse import urljoin
from dotenv import load_dotenv

class KaggleDiscussionDownloader:
    def __init__(self, competition_url, output_dir):
        self.base_url = "https://www.kaggle.com"
        self.competition_url = competition_url
        self.output_dir = output_dir
        self.session = requests.Session()
        load_dotenv('secrets/kaggle.env')
        self.username = os.getenv('KAGGLE_USERNAME')
        self.password = os.getenv('KAGGLE_PASSWORD')

    def login(self):
        login_url = urljoin(self.base_url, "/account/login")
        # Retrieve the login page to get CSRF token or any necessary hidden fields
        response = self.session.get(login_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']

        payload = {
            'username': self.username,
            'password': self.password,
            'csrfmiddlewaretoken': csrf_token
        }
        headers = {
            'Referer': login_url
        }
        post_response = self.session.post(login_url, data=payload, headers=headers)
        if post_response.status_code == 200 and "Logout" in post_response.text:
            print("Successfully logged in to Kaggle.")
        else:
            raise Exception("Failed to log in to Kaggle. Check your credentials.")

    def fetch_discussions(self):
        discussions = []
        page = 0
        while True:
            url = f"{self.competition_url}/pagination?page={page}&pageSize=20&sortBy=published"
            response = self.session.get(url)
            if response.status_code != 200:
                print(f"Failed to retrieve page {page}. Status code: {response.status_code}")
                break

            data = response.json()
            if not data['payload']['discussionList']:
                break

            for discussion in data['payload']['discussionList']:
                discussions.append({
                    'title': discussion['title'],
                    'author': discussion['author']['username'],
                    'published': discussion['published'],
                    'content': discussion['content']
                })

            print(f"Fetched page {page} with {len(data['payload']['discussionList'])} discussions.")
            page += 1
            time.sleep(1)  # Respectful delay between requests

        return discussions

    def save_discussions(self, discussions):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for idx, discussion in enumerate(discussions, start=1):
            filename = os.path.join(self.output_dir, f"discussion_{idx}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(discussion, f, ensure_ascii=False, indent=4)
            print(f"Saved {discussion['title']} to {filename}")

    def run(self):
        self.login()
        discussions = self.fetch_discussions()
        self.save_discussions(discussions)
        print(f"Total discussions downloaded: {len(discussions)}")

if __name__ == "__main__":
    COMPETITION_URL = "https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion"
    OUTPUT_DIR = "public/discussion/"

    downloader = KaggleDiscussionDownloader(COMPETITION_URL, OUTPUT_DIR)
    downloader.run()