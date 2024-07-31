import requests
import json
import os
from urllib.parse import unquote

# Define a User-Agent string
USER_AGENT = "WikimediaImageScraper/1.0 (https://example.com/your-app; yourname@example.com)"

def fetch_wikimedia_images(search_term, num_images=10):
    base_url = "https://commons.wikimedia.org/w/api.php"
    
    headers = {
        "User-Agent": USER_AGENT
    }
    
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": f"{search_term} filetype:bitmap",
        "srnamespace": "6",
        "srlimit": str(num_images),
    }
    
    response = requests.get(base_url, params=params, headers=headers)
    data = json.loads(response.text)
    
    image_titles = [item['title'] for item in data['query']['search']]
    
    image_urls = []
    for title in image_titles:
        file_params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "url",
            "titles": title,
        }
        file_response = requests.get(base_url, params=file_params, headers=headers)
        file_data = json.loads(file_response.text)
        
        page = next(iter(file_data['query']['pages'].values()))
        if 'imageinfo' in page:
            image_urls.append(page['imageinfo'][0]['url'])
    
    return image_urls

def download_images(urls, folder='data/downloaded_images/'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    headers = {
        "User-Agent": USER_AGENT
    }
    
    for url in urls:
        filename = unquote(url.split('/')[-1])
        filepath = os.path.join(folder, filename)
        
        with requests.get(url, stream=True, headers=headers) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded: {filename}")

# Example usage
search_term = "roman professional soldiers"
num_images = 5

image_urls = fetch_wikimedia_images(search_term, num_images)
download_images(image_urls,folder=f'data/downloaded_images/{search_term}/')