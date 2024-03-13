import scrapy


class ParssSpider(scrapy.Spider):
    name = "parss"
    allowed_domains = ["podcastaddict.com"]
    start_urls = ["https://podcastaddict.com"]


    def start_requests(self, lang="el"):
        urls = [
            f"https://podcastaddict.com/?lang={lang}",  # Replace with the website URL you want to scrape
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse_top_level)

    def parse_top_level(self, response):
        # Find the ul with class="caption" and content="Categories"
        ul = response.xpath("//ul[@class='list-unstyled']")
        # Extract all <a> elements inside that div
        for a in ul.xpath(".//a"):
            link_text = a.xpath("text()").get()
            link_url = a.xpath("@href").get()
            yield scrapy.Request(
                url=response.urljoin(link_url),
                callback=self.parse_podcatch,
                meta={"category": link_text},
            )

    def parse_podcatch(self, response):
        category = response.request.meta["category"]
        # Find the div with class="caption" and content="Categories"
        card_div = response.xpath("//div[@id='centertext']")
        # Extract all <a> elements inside that div
        for a in card_div.xpath(".//a"):
            name = a.xpath(".//h3").xpath("text()").get()
            author = a.xpath(".//p").xpath("text()").get()
            link_url = a.xpath("@href").get()
            meta = {
                "podcast_title": name,
                "author": author,
                "category": category,
            }
            yield scrapy.Request(
                url=response.urljoin(link_url), callback=self.parse, meta=meta
            )

    def parse(self, response):
        meta = response.request.meta
        link = response.xpath("//a[@id='copyRSSBtn']").xpath("@onclick").get()
        link = link.split("copyTopClipBoard('")[-1].split("',")[0]
        # reg = re.compile(r"copyTopClipBoard(.*)")
        # link = re.findall(reg, link)[0].split("',")[0]
        meta["rss"] = link
        yield meta
    def parse(self, response):
        pass
