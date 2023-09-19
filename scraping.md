# Sitemap Link Extractor

The provided Python script aims to extract all links from a given sitemap URL. This tool is designed with the capacity to handle sitemaps that may be divided across multiple files by following sitemap index links, a common approach for extensive websites.

## Key Components:

1. **Libraries & Modules**:
   - `xml.etree.ElementTree`: Utilized for XML parsing.
   - `requests`: Used to fetch the content of the sitemap from the provided URL.

2. **Main Functionality - `extract_links_from_sitemap`**:
   - **Input**: A sitemap URL (`sitemap_url`).
   - **Output**: A list of extracted URLs.
   - **Operation**: 
     - Fetches the content from the provided sitemap URL.
     - Dynamically determines the XML namespace of the sitemap.
     - Checks whether the sitemap is an index pointing to multiple sitemap files or a regular sitemap.
     - Extracts the links accordingly.

3. **Usage Block**:
   - Executes the link extraction process if the script is run directly.
   - Replace the placeholder sitemap URL (`"your_sitemap_url_here"`) with the actual sitemap URL to use the script.

## How to Use:

1. Ensure the required libraries (`requests`) are installed.
2. Replace the placeholder sitemap URL in the script with your desired sitemap's URL.
3. Execute the script. The extracted links will be printed to the console.

## Potential Enhancements:

While the script handles most typical sitemap structures, some potential enhancements and edge cases, like handling recursive sitemap indexes (a rare occurrence), are not currently addressed.
