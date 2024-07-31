import requests
import os
from urllib.parse import urlparse, unquote

class OpenMediaScraper:
    def __init__(self):
        self.apis = {
            'unsplash': {
                'url': 'https://api.unsplash.com/search/photos',
                'params': {
                    'query': '',
                    'per_page': 10,
                    'client_id': 'YOUR_UNSPLASH_API_KEY'
                },
                'result_key': 'results',
                'image_url_key': 'urls.regular'
            },
            'pexels': {
                'url': 'https://api.pexels.com/v1/search',
                'params': {
                    'query': '',
                    'per_page': 10
                },
                'headers': {
                    'Authorization': 'YOUR_PEXELS_API_KEY'
                },
                'result_key': 'photos',
                'image_url_key': 'src.large'
            },
            'pixabay': {
                'url': 'https://pixabay.com/api/',
                'params': {
                    'q': '',
                    'per_page': 10,
                    'key': 'YOUR_PIXABAY_API_KEY'
                },
                'result_key': 'hits',
                'image_url_key': 'largeImageURL'
            }
        }

    def fetch_images(self, site, query, num_images=10):
        if site not in self.apis:
            raise ValueError(f"Unsupported site: {site}")

        api_info = self.apis[site]
        api_info['params']['query' if 'query' in api_info['params'] else 'q'] = query
        api_info['params']['per_page'] = num_images

        response = requests.get(api_info['url'], params=api_info['params'], headers=api_info.get('headers', {}))
        response.raise_for_status()
        data = response.json()

        results = data[api_info['result_key']]
        image_urls = []

        for result in results:
            keys = api_info['image_url_key'].split('.')
            url = result
            for key in keys:
                url = url[key]
            image_urls.append(url)

        return image_urls

    def download_images(self, urls, folder='downloaded_images'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for url in urls:
            filename = unquote(os.path.basename(urlparse(url).path))
            filepath = os.path.join(folder, filename)

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded: {filename}")

# Example usage
scraper = OpenMediaScraper()
search_term = "cats"
num_images = 5

for site in ['unsplash', 'pexels', 'pixabay']:
    print(f"\nFetching images from {site}...")
    try:
        image_urls = scraper.fetch_images(site, search_term, num_images)
        scraper.download_images(image_urls, folder=f'downloaded_images_{site}')
    except Exception as e:
        print(f"Error fetching images from {site}: {str(e)}")