import asyncio
import tempfile
import uuid
import os
from typing import Dict, List
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
from bs4 import BeautifulSoup
import streamlit as st

class WebScraper:
    def __init__(self, headless: bool = True, browser_type: str = "chromium"):
        self.headless = headless
        self.browser_type = browser_type
        self.last_scraped_file = None
        self.name = "web_scraper"
        self.description = "Scrapes the content of a web page and returns structured data including titles, links, and content."
        self.parameters = {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the web page to scrape."
                }
            },
            "required": ["url"]
        }
        self.instructions = """Using this scraped data, create a structured JSON response that includes the most relevant and important information from the website.
Focus on the main content. Do not include HTML tags or unnecessary details.
Ensure your response is in valid JSON format without any additional text or comments."""

    async def scrape_page(self, url: str) -> str:
        async with async_playwright() as p:
            browser = await getattr(p, self.browser_type).launch(headless=self.headless)
            context = await browser.new_context()
            page = await context.new_page()

            try:
                await stealth_async(page)
                await page.goto(url)
                html_content = await page.content()
                
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', prefix=f'{uuid.uuid4()}_') as temp_file:
                    temp_file.write(html_content)
                    self.last_scraped_file = temp_file.name
                
            except Exception as e:
                st.error(f"Error scraping page: {e}")
                html_content = ""
            finally:
                await browser.close()

        return html_content

    @staticmethod
    def extract_titles_articles_links(raw_html: str) -> List[Dict[str, str]]:
        soup = BeautifulSoup(raw_html, 'html.parser')
        extracted_data = []
        
        main_content = soup.find('main') or soup.find('body')
        
        for article in main_content.find_all(['article', 'section', 'div'], recursive=False):
            title_tag = article.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            link_tags = article.find_all('a', href=True)
            content = article.get_text(separator="\n", strip=True)
            
            if title_tag and content:
                extracted_data.append({
                    'title': title_tag.get_text(strip=True),
                    'links': [{'text': link.get_text(strip=True), 'href': link['href']} for link in link_tags],
                    'content': content[:1000]  # Limit content to 1000 characters
                })
        
        return extracted_data

    async def query_page_content(self, url: str) -> Dict[str, any]:
        raw_html = await self.scrape_page(url)
        extracted_data = self.extract_titles_articles_links(raw_html)
        return {
            "url": url,
            "extracted_data": extracted_data,
            "instructions": self.instructions
        }
