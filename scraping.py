import xml.etree.ElementTree as ET
import requests

def extract_links_from_sitemap(sitemap_url):
    """
    Extract all links from a given sitemap URL. This function can handle
    sitemaps that are split across multiple files by following sitemap index links.

    Parameters:
    - sitemap_url (str): The URL of the root sitemap or sitemap index.

    Returns:
    - list[str]: A list of extracted links from the sitemap.
    """
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Fetch the sitemap content from the provided URL
    response = requests.get(sitemap_url, headers=headers)
    root = ET.fromstring(response.content)

    # Determine the XML namespace dynamically
    namespace = {'sitemap': root.tag.split('}')[0].strip('{')}

    links = []

    # Check if the provided sitemap is an index of multiple sitemaps
    if root.tag.endswith('sitemapindex'):
        for sitemap in root.findall('sitemap:sitemap', namespace):
            loc = sitemap.find('sitemap:loc', namespace)
            if loc is not None:
                # Recursively extract links from the individual sitemap
                links.extend(extract_links_from_sitemap(loc.text))
    else:
        # Extract links from the current sitemap
        for url in root.findall('sitemap:url', namespace):
            loc = url.find('sitemap:loc', namespace)
            if loc is not None:
                links.append(loc.text)

    return sorted(links)

# Usage
if __name__ == "__main__":
    sitemap_url = "your_sitemap_url_here"
    links = extract_links_from_sitemap(sitemap_url)
